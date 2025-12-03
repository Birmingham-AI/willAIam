import React, { useState, useEffect } from 'react';
import { Upload, Loader2, CheckCircle, XCircle, Youtube, ChevronDown, ChevronUp, RefreshCw, Trash2 } from 'lucide-react';
import config from '../../config';

interface JobStatus {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  message: string;
  video_id?: string;
  chunk_count?: number;
  error?: string;
}

interface Source {
  id: string;
  source_id: string;
  session_info: string;
  chunk_count: number;
  processed_at: string;
}

const YouTubeUpload: React.FC = () => {
  const [url, setUrl] = useState('');
  const [sessionInfo, setSessionInfo] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [chunkSize, setChunkSize] = useState(1000);
  const [overlap, setOverlap] = useState(1);
  const [language, setLanguage] = useState('en');

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeJobs, setActiveJobs] = useState<JobStatus[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [sources, setSources] = useState<Source[]>([]);
  const [loadingSources, setLoadingSources] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Fetch existing sources on mount
  useEffect(() => {
    fetchSources();
  }, []);

  // Poll for job status when there are processing jobs
  useEffect(() => {
    const processingJobs = activeJobs.filter(job => job.status === 'processing');
    if (processingJobs.length === 0) return;

    const pollInterval = setInterval(async () => {
      const updates = await Promise.all(
        processingJobs.map(async (job) => {
          try {
            const response = await fetch(`${config.apiBaseUrl}/api/youtube/status/${job.job_id}`);
            if (response.ok) {
              return await response.json() as JobStatus;
            }
          } catch (err) {
            console.error(`Failed to poll job ${job.job_id}:`, err);
          }
          return null;
        })
      );

      // Check if any jobs completed
      const hasNewlyCompleted = updates.some(
        (update, i) => update && update.status === 'completed' && processingJobs[i].status === 'processing'
      );

      // Update jobs state with new statuses
      setActiveJobs(prev => prev.map(job => {
        const update = updates.find((u) => u && u.job_id === job.job_id);
        return update || job;
      }));

      if (hasNewlyCompleted) {
        fetchSources();
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [activeJobs]);

  const fetchSources = async () => {
    try {
      setLoadingSources(true);
      const response = await fetch(`${config.apiBaseUrl}/api/youtube/sources`);
      if (response.ok) {
        const data = await response.json();
        setSources(data.sources || []);
      }
    } catch (err) {
      console.error('Failed to fetch sources:', err);
    } finally {
      setLoadingSources(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const response = await fetch(`${config.apiBaseUrl}/api/youtube/upload`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url,
          session_info: sessionInfo,
          chunk_size: chunkSize,
          overlap,
          language,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start upload');
      }

      const data = await response.json();
      // Add new job to the array instead of replacing
      setActiveJobs(prev => [...prev, {
        job_id: data.job_id,
        status: 'processing',
        message: data.message,
        video_id: data.video_id,
      }]);

      // Clear form on successful submission
      setUrl('');
      setSessionInfo('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  const deleteSource = async (id: string) => {
    if (!confirm('Are you sure you want to delete this video and all its embeddings?')) {
      return;
    }

    setDeletingId(id);
    try {
      const response = await fetch(`${config.apiBaseUrl}/api/youtube/sources/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete source');
      }

      // Remove from local state
      setSources(sources.filter(s => s.id !== id));
    } catch (err) {
      console.error('Failed to delete source:', err);
      alert(err instanceof Error ? err.message : 'Failed to delete source');
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const dismissJob = (jobId: string) => {
    setActiveJobs(prev => prev.filter(job => job.job_id !== jobId));
  };

  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 p-6 overflow-hidden flex flex-col">
      {/* Header - Centered */}
      <div className="text-center mb-6 flex-shrink-0">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
          <Youtube className="w-8 h-8 text-red-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-800">YouTube Transcription</h1>
        <p className="text-gray-600 mt-2">Add YouTube videos to the knowledge base</p>
      </div>

      {/* 2-Column Layout */}
      <div className="flex-1 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-2 gap-6 overflow-hidden">
        {/* Left Column - Upload Form & Status */}
        <div className="flex flex-col overflow-y-auto">
          {/* Upload Form */}
          <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
            <form onSubmit={handleSubmit}>
              {/* URL Input */}
              <div className="mb-4">
                <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-1">
                  YouTube URL *
                </label>
                <input
                  type="text"
                  id="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                  required
                />
              </div>

              {/* Session Info Input */}
              <div className="mb-4">
                <label htmlFor="sessionInfo" className="block text-sm font-medium text-gray-700 mb-1">
                  Session Info *
                </label>
                <input
                  type="text"
                  id="sessionInfo"
                  value={sessionInfo}
                  onChange={(e) => setSessionInfo(e.target.value)}
                  placeholder="e.g., Nov 2024 Birmingham AI Meetup"
                  className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                  required
                />
              </div>

              {/* Advanced Options Toggle */}
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-800 mb-4"
              >
                {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                Advanced Options
              </button>

              {/* Advanced Options */}
              {showAdvanced && (
                <div className="bg-gray-50 rounded-xl p-4 mb-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="chunkSize" className="block text-sm font-medium text-gray-700 mb-1">
                        Chunk Size (chars)
                      </label>
                      <input
                        type="number"
                        id="chunkSize"
                        value={chunkSize}
                        onChange={(e) => setChunkSize(parseInt(e.target.value) || 1000)}
                        min={100}
                        max={5000}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                      />
                    </div>
                    <div>
                      <label htmlFor="overlap" className="block text-sm font-medium text-gray-700 mb-1">
                        Overlap (sentences)
                      </label>
                      <input
                        type="number"
                        id="overlap"
                        value={overlap}
                        onChange={(e) => setOverlap(parseInt(e.target.value) || 1)}
                        min={0}
                        max={10}
                        className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                      />
                    </div>
                  </div>
                  <div>
                    <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                      Language Code
                    </label>
                    <input
                      type="text"
                      id="language"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                      placeholder="en"
                      className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                    />
                  </div>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3">
                  <XCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isSubmitting || !url || !sessionInfo}
                className="w-full py-3 px-4 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Process Video
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Active Jobs Status */}
          {activeJobs.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-700">
                Active Jobs ({activeJobs.filter(j => j.status === 'processing').length} processing)
              </h3>
              {activeJobs.map((job) => (
                <div
                  key={job.job_id}
                  className={`bg-white rounded-2xl shadow-xl p-4 border-l-4 ${
                    job.status === 'processing' ? 'border-blue-500' :
                    job.status === 'completed' ? 'border-green-500' :
                    'border-red-500'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {job.status === 'processing' && (
                      <Loader2 className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0" />
                    )}
                    {job.status === 'completed' && (
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                    )}
                    {job.status === 'failed' && (
                      <XCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-gray-800 text-sm">
                          {job.status === 'processing' ? 'Processing...' :
                           job.status === 'completed' ? 'Completed!' :
                           'Failed'}
                        </h4>
                        {job.status !== 'processing' && (
                          <button
                            onClick={() => dismissJob(job.job_id)}
                            className="text-gray-400 hover:text-gray-600 p-1"
                            title="Dismiss"
                          >
                            <XCircle className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                      <p className="text-gray-600 text-xs mt-1 truncate">{job.message}</p>
                      {job.video_id && (
                        <p className="text-gray-500 text-xs mt-1">Video: {job.video_id}</p>
                      )}
                      {job.chunk_count !== undefined && (
                        <p className="text-gray-500 text-xs">Chunks: {job.chunk_count}</p>
                      )}
                      {job.error && (
                        <p className="text-red-600 text-xs mt-1">{job.error}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column - Processed Videos */}
        <div className="bg-white rounded-2xl shadow-xl p-6 flex flex-col h-full overflow-hidden">
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            <h2 className="text-lg font-semibold text-gray-800">Processed Videos</h2>
            <button
              onClick={fetchSources}
              disabled={loadingSources}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh"
            >
              <RefreshCw className={`w-5 h-5 ${loadingSources ? 'animate-spin' : ''}`} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto">
            {loadingSources ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-6 h-6 text-gray-400 animate-spin" />
              </div>
            ) : sources.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Youtube className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No videos processed yet</p>
              </div>
            ) : (
              <div className="space-y-3 pr-2">
                {sources.map((source) => (
                  <div
                    key={source.id}
                    className="flex items-center gap-4 p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors"
                  >
                    <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Youtube className="w-5 h-5 text-red-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-800 truncate">{source.session_info}</p>
                      <div className="flex items-center gap-3 text-xs text-gray-500 mt-1">
                        <span>{source.source_id}</span>
                        <span>{source.chunk_count} chunks</span>
                        <span>{formatDate(source.processed_at)}</span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <a
                        href={`https://www.youtube.com/watch?v=${source.source_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                      >
                        View
                      </a>
                      <button
                        onClick={() => deleteSource(source.id)}
                        disabled={deletingId === source.id}
                        className="p-1.5 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                        title="Delete"
                      >
                        {deletingId === source.id ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Trash2 className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default YouTubeUpload;
