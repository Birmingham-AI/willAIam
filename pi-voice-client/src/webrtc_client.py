"""
WebRTC Client for connecting to OpenAI Realtime API.

Establishes WebRTC connection and handles audio streaming and events.
"""

import logging
import json
import asyncio
import struct
import time
from math import gcd
import numpy as np
from typing import Optional, Callable
import httpx
from scipy import signal
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import AudioFrame as AVAudioFrame
from .config import config

logger = logging.getLogger(__name__)


class MicrophoneAudioTrack(MediaStreamTrack):
    """Audio track that streams microphone input from PyAudio."""

    kind = "audio"

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        """
        Initialize microphone audio track.

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self._started = False
        self._pts = 0  # Presentation timestamp counter

    def add_audio_data(self, audio_data: bytes) -> None:
        """
        Add audio data to be sent.

        Args:
            audio_data: PCM audio data (16-bit, mono)
        """
        if self._started:
            self.audio_queue.put_nowait(audio_data)

    async def recv(self) -> Optional[AVAudioFrame]:
        """
        Receive audio frame to send through WebRTC.

        Returns:
            AudioFrame for WebRTC transmission
        """
        try:
            # Get audio data from queue
            audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)

            # Convert bytes to numpy array
            # audio_data is 16-bit PCM, mono
            num_samples = len(audio_data) // 2
            samples = np.frombuffer(audio_data, dtype=np.int16)

            # Create AV AudioFrame from numpy array
            # from_ndarray expects shape (channels, samples) for multi-channel
            # or (samples,) for mono, but we need to ensure it's a 2D array
            if self.channels == 1:
                samples_array = samples.reshape(1, -1)  # Shape: (1, num_samples)
            else:
                # For stereo, reshape appropriately
                samples_array = samples.reshape(self.channels, -1)

            frame = AVAudioFrame.from_ndarray(
                samples_array, format="s16", layout="mono" if self.channels == 1 else "stereo"
            )
            frame.rate = self.sample_rate
            frame.pts = self._pts
            self._pts += num_samples  # Increment PTS by number of samples

            return frame

        except asyncio.TimeoutError:
            # Return silence if no data available
            num_samples = self.sample_rate // 10  # 100ms of silence
            silence_samples = np.zeros(num_samples, dtype=np.int16)
            samples_array = silence_samples.reshape(1, -1)  # Shape: (1, num_samples)
            
            frame = AVAudioFrame.from_ndarray(
                samples_array, format="s16", layout="mono" if self.channels == 1 else "stereo"
            )
            frame.rate = self.sample_rate
            frame.pts = self._pts
            self._pts += num_samples
            return frame
        except Exception as e:
            logger.error(f"Error in microphone track recv: {e}")
            return None


class WebRTCClient:
    """WebRTC client for OpenAI Realtime API."""

    def __init__(
        self,
        on_audio_received: Optional[Callable[[bytes], None]] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        """
        Initialize WebRTC client.

        Args:
            on_audio_received: Callback for received audio chunks
            on_event: Callback for Realtime API events
        """
        self.api_base_url = config.API_BASE_URL
        self.on_audio_received = on_audio_received
        self.on_event = on_event

        self.peer_connection: Optional[RTCPeerConnection] = None
        self.data_channel = None
        self.microphone_track: Optional[MicrophoneAudioTrack] = None
        self.ephemeral_key: Optional[str] = None
        self.is_connected = False
        self.audio_receive_task: Optional[asyncio.Task] = None

    async def get_ephemeral_token(self) -> str:
        """
        Get ephemeral token from backend.

        Returns:
            Ephemeral key for OpenAI API
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/v1/realtime/session",
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code != 200:
                    raise Exception(
                        f"Failed to create session: {response.status_code} - {response.text}"
                    )

                data = response.json()
                ephemeral_key = data.get("client_secret", {}).get("value")

                if not ephemeral_key:
                    raise Exception("No ephemeral key received from backend")

                logger.info("Got ephemeral token from backend")
                return ephemeral_key

        except Exception as e:
            logger.error(f"Error getting ephemeral token: {e}")
            raise

    async def connect(self) -> None:
        """Establish WebRTC connection to OpenAI Realtime API."""
        try:
            # Ensure clean state before starting new connection
            # Cancel any existing audio receive task from previous connection
            if self.audio_receive_task:
                logger.warning("Found existing audio_receive_task, cancelling before new connection")
                self.audio_receive_task.cancel()
                try:
                    await self.audio_receive_task
                except (asyncio.CancelledError, Exception):
                    pass
                self.audio_receive_task = None
            
            # Reset connection state
            self.is_connected = False
            
            # Get ephemeral token
            self.ephemeral_key = await self.get_ephemeral_token()

            # Create peer connection
            self.peer_connection = RTCPeerConnection()

            # Create microphone track
            self.microphone_track = MicrophoneAudioTrack(
                sample_rate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
            )
            self.microphone_track._started = True
            self.peer_connection.addTrack(self.microphone_track)

            # Handle incoming audio track
            @self.peer_connection.on("track")
            def on_track(track: MediaStreamTrack):
                logger.info(f"Received track: {track.kind}, id: {track.id}")
                if track.kind == "audio":
                    # Only start audio receive task if connection is established and task doesn't exist
                    if self.is_connected and not self.audio_receive_task:
                        logger.info("Starting audio receive task")
                        self.audio_receive_task = asyncio.create_task(
                            self._handle_incoming_audio(track)
                        )
                    elif self.audio_receive_task:
                        logger.warning("Audio receive task already exists, ignoring new track")
                    else:
                        logger.warning("Connection not yet established, will start audio handler after connection")
                else:
                    logger.warning(f"Received non-audio track: {track.kind}")

            # Create data channel for events
            self.data_channel = self.peer_connection.createDataChannel("oai-events")
            self.data_channel.on("message", self._handle_data_channel_message)

            # Create offer
            offer = await self.peer_connection.createOffer()
            await self.peer_connection.setLocalDescription(offer)

            # Exchange SDP with OpenAI
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/realtime?model=gpt-realtime",
                    headers={
                        "Authorization": f"Bearer {self.ephemeral_key}",
                        "Content-Type": "application/sdp",
                    },
                    content=offer.sdp,
                )

                # OpenAI returns 201 (Created) for successful SDP exchange, not 200
                if response.status_code not in [200, 201]:
                    raise Exception(
                        f"Failed to establish WebRTC connection: {response.status_code} - {response.text}"
                    )

                answer_sdp = response.text
                await self.peer_connection.setRemoteDescription(
                    RTCSessionDescription(sdp=answer_sdp, type="answer")
                )

            # Set connection state before checking for tracks
            self.is_connected = True
            logger.info("WebRTC connection established")
            
            # Wait a brief moment for tracks to be established
            await asyncio.sleep(0.1)
            
            # Check for existing tracks (they might be available immediately)
            receivers = self.peer_connection.getReceivers()
            logger.info(f"Connection has {len(receivers)} receivers, {len(self.peer_connection.getSenders())} senders")
            for receiver in receivers:
                if receiver.track:
                    logger.info(f"Found existing receiver track: {receiver.track.kind}, id: {receiver.track.id}")
                    if receiver.track.kind == "audio" and not self.audio_receive_task:
                        logger.info("Starting audio receive task for existing track")
                        self.audio_receive_task = asyncio.create_task(
                            self._handle_incoming_audio(receiver.track)
                        )

        except Exception as e:
            logger.error(f"Error establishing WebRTC connection: {e}")
            await self.cleanup()
            raise

    def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to OpenAI.

        Args:
            audio_data: PCM audio data (16-bit, mono)
        """
        if self.microphone_track and self.is_connected:
            self.microphone_track.add_audio_data(audio_data)

    async def _handle_incoming_audio(self, track: MediaStreamTrack) -> None:
        """
        Handle incoming audio from OpenAI.

        Args:
            track: Audio track from peer connection
        """
        logger.info("Audio receive handler started")
        try:
            frame_count = 0
            timeout_count = 0
            first_frame_timeout = 5.0  # Wait up to 5 seconds for first frame
            subsequent_timeout = 1.0  # 1 second timeout for subsequent frames
            
            # Wait for connection to be fully established
            if not self.is_connected:
                logger.warning("Connection not yet established, waiting...")
                wait_count = 0
                while not self.is_connected and wait_count < 10:
                    await asyncio.sleep(0.1)
                    wait_count += 1
                if not self.is_connected:
                    logger.error("Connection not established after waiting, exiting audio handler")
                    return
            
            while self.is_connected:
                try:
                    # Use longer timeout for first frame (audio may take time to start)
                    timeout = first_frame_timeout if frame_count == 0 else subsequent_timeout
                    # Wait for frame with timeout to allow checking connection status
                    frame = await asyncio.wait_for(track.recv(), timeout=timeout)
                    timeout_count = 0  # Reset timeout counter on successful receive
                    first_frame_timeout = subsequent_timeout  # Use shorter timeout after first frame
                    
                    if frame:
                        frame_count += 1
                        if frame_count == 1:
                            logger.info(
                                f"Received first audio frame from OpenAI: "
                                f"sample_rate={frame.sample_rate}Hz, "
                                f"samples={frame.samples}, "
                                f"format={frame.format}, "
                                f"layout={frame.layout}"
                            )
                            logger.info(
                                f"Output stream configured for: {config.AUDIO_SAMPLE_RATE}Hz, "
                                f"channels: {config.AUDIO_CHANNELS}"
                            )
                            # Check for sample rate mismatch
                            if frame.sample_rate != config.AUDIO_SAMPLE_RATE:
                                logger.warning(
                                    f"Sample rate mismatch detected! "
                                    f"Incoming: {frame.sample_rate}Hz, "
                                    f"Output configured: {config.AUDIO_SAMPLE_RATE}Hz. "
                                    f"Will resample to match output rate."
                                )
                            else:
                                logger.info("Sample rates match - no resampling needed")
                        elif frame_count % 250 == 0:  # Log every 250 frames (less verbose)
                            logger.info(f"Received {frame_count} audio frames from OpenAI")
                        
                        if self.on_audio_received:
                            try:
                                # Convert AudioFrame to bytes (PCM 16-bit)
                                # frame is an av.AudioFrame
                                import numpy as np

                                # Convert to numpy array
                                # PyAV's to_ndarray() returns the data in the frame's native format
                                array = frame.to_ndarray()
                                
                                # Check and convert data type if needed
                                if array.dtype == np.float32 or array.dtype == np.float64:
                                    # Float audio is normalized -1.0 to 1.0, convert to int16
                                    array = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
                                    if frame_count == 1:
                                        logger.info(f"Converted float{array.dtype.itemsize * 8} to int16")
                                elif array.dtype != np.int16:
                                    # Other format - try to convert directly
                                    logger.warning(f"Unexpected audio dtype {array.dtype}, converting to int16")
                                    array = array.astype(np.int16)
                                
                                # Get frame layout information
                                is_stereo = False
                                num_channels = 1
                                try:
                                    if hasattr(frame.layout, 'name'):
                                        is_stereo = frame.layout.name == 'stereo'
                                    if hasattr(frame.layout, 'channels'):
                                        num_channels = len(frame.layout.channels)
                                    elif hasattr(frame.layout, 'channel_count'):
                                        num_channels = frame.layout.channel_count
                                except:
                                    pass
                                
                                # Log frame properties
                                if frame_count == 1:
                                    logger.info(
                                        f"Frame: layout={frame.layout}, is_stereo={is_stereo}, "
                                        f"num_channels={num_channels}, frame.samples={frame.samples}, "
                                        f"array_shape={array.shape}, dtype={array.dtype}, "
                                        f"min={array.min()}, max={array.max()}"
                                    )
                                
                                # Reshape array if needed
                                # PyAV should return (channels, samples) for planar format
                                # But sometimes it returns (1, total_samples) for interleaved
                                if is_stereo and len(array.shape) == 2 and array.shape[0] == 1:
                                    # Likely interleaved stereo in shape (1, total_samples)
                                    # Reshape to planar: (2, samples_per_channel)
                                    total_samples = array.shape[1]
                                    samples_per_channel = total_samples // 2
                                    if total_samples == frame.samples * 2:
                                        # Reshape to planar format
                                        array = array.reshape(2, samples_per_channel)
                                        if frame_count == 1:
                                            logger.info(
                                                f"Reshaped interleaved stereo: (1, {total_samples}) -> (2, {samples_per_channel})"
                                            )
                                    elif frame_count == 1:
                                        logger.warning(
                                            f"Unexpected array shape for stereo: {array.shape}, "
                                            f"frame.samples={frame.samples}"
                                        )

                                # Handle different data types and channel layouts
                                if array.dtype == np.float32 or array.dtype == np.float64:
                                    # Normalized float audio (-1.0 to 1.0), scale to int16
                                    # Clip values to prevent overflow
                                    array = np.clip(array, -1.0, 1.0)
                                    
                                    # Convert stereo to mono if needed
                                    if is_stereo and config.AUDIO_CHANNELS == 1:
                                        if len(array.shape) == 2 and array.shape[0] == 2:
                                            # Planar stereo (2, samples): average channels
                                            audio_array = ((array[0] + array[1]) / 2.0 * 32767).astype(np.int16)
                                        elif len(array.shape) == 1:
                                            # Interleaved stereo: take every other sample (left channel)
                                            audio_array = (array[::2] * 32767).astype(np.int16)
                                        else:
                                            # Unknown format, take first channel
                                            audio_array = (array[0] * 32767).astype(np.int16)
                                    elif len(array.shape) == 2:
                                        # Multi-channel: take first channel
                                        audio_array = (array[0] * 32767).astype(np.int16)
                                    else:
                                        # Mono
                                        audio_array = (array * 32767).astype(np.int16)
                                    
                                    if frame_count == 1:
                                        logger.info(f"Converted float32 audio to int16 (shape: {audio_array.shape})")
                                elif array.dtype == np.int16:
                                    # Already int16, use directly
                                    
                                    # Convert stereo to mono if needed
                                    if is_stereo and config.AUDIO_CHANNELS == 1:
                                        if len(array.shape) == 2 and array.shape[0] == 2:
                                            # Planar stereo (2, samples): average channels for better quality
                                            audio_array = ((array[0].astype(np.int32) + array[1].astype(np.int32)) // 2).astype(np.int16)
                                            if frame_count == 1:
                                                logger.info(
                                                    f"Converted planar stereo to mono (averaged): "
                                                    f"{array.shape[1]} samples per channel -> {len(audio_array)} samples"
                                                )
                                        elif len(array.shape) == 1:
                                            # 1D array - interleaved stereo
                                            if len(array) == frame.samples * 2:
                                                # Interleaved: take left channel (every other sample starting at 0)
                                                audio_array = array[::2]
                                                if frame_count == 1:
                                                    logger.info(
                                                        f"De-interleaved 1D stereo: {len(array)} samples -> {len(audio_array)} samples"
                                                    )
                                            else:
                                                # Already mono or unexpected
                                                audio_array = array
                                                if frame_count == 1:
                                                    logger.warning(
                                                        f"Unexpected 1D array length: {len(array)}, "
                                                        f"expected {frame.samples * 2} for stereo"
                                                    )
                                        else:
                                            # Fallback: flatten and take first channel worth of samples
                                            logger.warning(f"Unexpected array shape {array.shape} for stereo, using fallback")
                                            flat = array.flatten()
                                            audio_array = flat[:frame.samples] if len(flat) >= frame.samples else flat
                                    elif len(array.shape) == 2:
                                        # Multi-channel: take first channel
                                        audio_array = array[0]
                                    else:
                                        # Mono
                                        audio_array = array
                                    
                                    if frame_count == 1:
                                        logger.info(
                                            f"Final audio array: shape={audio_array.shape}, "
                                            f"length={len(audio_array)}, min={audio_array.min()}, "
                                            f"max={audio_array.max()}, samples={frame.samples}"
                                        )
                                else:
                                    # Try to convert to int16
                                    logger.warning(f"Unexpected dtype {array.dtype}, attempting conversion to int16")
                                    if is_stereo and config.AUDIO_CHANNELS == 1:
                                        if len(array.shape) == 2 and array.shape[0] == 2:
                                            audio_array = ((array[0].astype(np.int32) + array[1].astype(np.int32)) // 2).astype(np.int16)
                                        elif len(array.shape) == 1:
                                            audio_array = array[::2].astype(np.int16)
                                        else:
                                            audio_array = array[0].astype(np.int16) if len(array.shape) == 2 else array.astype(np.int16)
                                    elif len(array.shape) == 2:
                                        audio_array = array[0].astype(np.int16)
                                    else:
                                        audio_array = array.astype(np.int16)

                                # Resample if sample rate mismatch
                                if frame.sample_rate != config.AUDIO_SAMPLE_RATE:
                                    original_samples = len(audio_array)
                                    target_rate = config.AUDIO_SAMPLE_RATE
                                    source_rate = frame.sample_rate
                                    
                                    # Use resample_poly for better real-time performance and quality
                                    # Calculate the up/down ratio using GCD for simplest ratio
                                    g = gcd(int(target_rate), int(source_rate))
                                    up = int(target_rate // g)
                                    down = int(source_rate // g)
                                    
                                    if frame_count == 1:
                                        logger.info(
                                            f"Resampling: {source_rate}Hz -> {target_rate}Hz "
                                            f"(ratio {up}:{down}, {original_samples} samples -> "
                                        )
                                    
                                    # Resample using polyphase method (better for real-time)
                                    audio_array = signal.resample_poly(audio_array, up, down).astype(np.int16)
                                    
                                    if frame_count == 1:
                                        logger.info(
                                            f"Resampled audio: {original_samples} -> {len(audio_array)} samples"
                                        )
                                else:
                                    if frame_count == 1:
                                        logger.info(
                                            f"No resampling needed: frame rate {frame.sample_rate}Hz matches output rate {config.AUDIO_SAMPLE_RATE}Hz"
                                        )
                                
                                # Convert to bytes
                                audio_data = audio_array.tobytes()
                                
                                # Check audio quality periodically (not just first frame)
                                if frame_count == 1:
                                    # Check if audio contains actual sound
                                    non_zero_samples = np.count_nonzero(audio_array)
                                    max_amplitude = np.abs(audio_array).max()
                                    # Only log if there's actual audio content (skip silence)
                                    if non_zero_samples > 0 or max_amplitude > 0:
                                        logger.info(
                                            f"First frame: {len(audio_data)} bytes, "
                                            f"non-zero: {non_zero_samples}/{len(audio_array)}, "
                                            f"max_amp: {max_amplitude}"
                                        )
                                    else:
                                        logger.warning("First audio frame is all zeros (silence) - OpenAI may not have generated response")
                                    
                                    if max_amplitude > 0 and max_amplitude < 100:
                                        logger.warning(f"Audio amplitude very low (max: {max_amplitude})")
                                elif frame_count == 10:
                                    # Check 10th frame to verify audio is actually coming through
                                    non_zero_samples = np.count_nonzero(audio_array)
                                    max_amplitude = np.abs(audio_array).max()
                                    if non_zero_samples == 0:
                                        logger.error("ERROR: 10th frame is still all zeros - no audio received!")
                                    elif max_amplitude < 100:
                                        logger.warning(f"10th frame amplitude still low (max: {max_amplitude})")
                                    else:
                                        logger.info(f"10th frame has audio: max_amp={max_amplitude}, non_zero={non_zero_samples}/{len(audio_array)}")
                                
                                self.on_audio_received(audio_data)
                            except Exception as e:
                                logger.error(f"Error processing audio frame {frame_count}: {e}", exc_info=True)
                        else:
                            logger.warning("on_audio_received callback not set")
                    else:
                        logger.debug("Received None frame from track")
                except asyncio.TimeoutError:
                    timeout_count += 1
                    # Wait longer before giving up - audio might still be coming
                    max_timeouts = 10 if frame_count == 0 else 5  # Changed from 3 to 5
                    
                    if timeout_count >= max_timeouts:
                        if frame_count == 0:
                            logger.error(
                                f"No audio frames received after {timeout_count * timeout:.1f} seconds. "
                                f"Connection: {self.is_connected}. Exiting handler."
                            )
                        else:
                            logger.info(
                                f"No audio frames for {timeout_count} seconds after {frame_count} frames. "
                                f"Stream complete. Exiting handler."
                            )
                        break
                    
                    # Log periodically, more frequently for first frame wait
                    log_interval = 5 if frame_count == 0 else 10
                    if timeout_count % log_interval == 0:
                        logger.debug(
                            f"No audio frames received for {timeout_count * timeout:.1f} seconds "
                            f"(connection: {self.is_connected}, frames received: {frame_count})"
                        )
                    continue
        except asyncio.CancelledError:
            logger.info("Audio receive handler cancelled")
        except Exception as e:
            logger.error(f"Error handling incoming audio: {e}", exc_info=True)
        finally:
            if frame_count == 0:
                logger.warning(
                    f"Audio receive handler finished without processing any frames. "
                    f"Connection state: {self.is_connected}"
                )
            else:
                logger.info(f"Audio receive handler finished after processing {frame_count} frames")

    def _handle_data_channel_message(self, message: str) -> None:
        """
        Handle messages from data channel.

        Args:
            message: JSON message string
        """
        try:
            event = json.loads(message)
            event_type = event.get("type", "")

            logger.debug(f"Received event: {event_type}")

            if self.on_event:
                self.on_event(event)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse data channel message: {e}")
        except Exception as e:
            logger.error(f"Error handling data channel message: {e}")

    async def send_event(self, event: dict) -> None:
        """
        Send event through data channel.

        Args:
            event: Event dictionary to send
        """
        if self.data_channel:
            state = self.data_channel.readyState
            logger.debug(f"Data channel state: {state}, event type: {event.get('type', 'unknown')}")
            if state == "open":
                try:
                    event_json = json.dumps(event)
                    self.data_channel.send(event_json)
                    logger.debug(f"Sent event: {event.get('type', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error sending event: {e}")
            else:
                logger.warning(f"Data channel not open (state: {state}), cannot send event: {event.get('type', 'unknown')}")
        else:
            logger.warning("Data channel not available, cannot send event")

    async def cleanup(self) -> None:
        """Clean up WebRTC connection."""
        try:
            # Set connection state to False FIRST to stop handlers
            self.is_connected = False
            
            # Cancel and cleanup audio receive task immediately
            if self.audio_receive_task:
                if not self.audio_receive_task.done():
                    self.audio_receive_task.cancel()
                    # Don't wait for it - just cancel and move on
                self.audio_receive_task = None
            
            # Close data channel
            if self.data_channel:
                try:
                    self.data_channel.close()
                except:
                    pass
                self.data_channel = None
            
            # Stop microphone track
            if self.microphone_track:
                self.microphone_track._started = False
                self.microphone_track = None
            
            # Close peer connection
            if self.peer_connection:
                try:
                    await asyncio.wait_for(self.peer_connection.close(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Peer connection close timed out")
                except Exception as e:
                    logger.debug(f"Error closing peer connection: {e}")
                self.peer_connection = None
            
            self.ephemeral_key = None
            logger.info("WebRTC connection cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up WebRTC connection: {e}")

