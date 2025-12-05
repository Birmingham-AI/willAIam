import React, { useState } from 'react';
import { ThumbsUp, ThumbsDown, Check } from 'lucide-react';
import apiService from '../../services/ApiService';

interface FeedbackControlsProps {
  traceId: string;
  isStreaming: boolean;
}

/**
 * Like/dislike feedback buttons for assistant messages
 */
const FeedbackControls: React.FC<FeedbackControlsProps> = ({ traceId, isStreaming }) => {
  const [selectedRating, setSelectedRating] = useState<'like' | 'dislike' | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedback = async (rating: 'like' | 'dislike') => {
    if (feedbackSubmitted || isSubmitting) return;

    setIsSubmitting(true);
    setSelectedRating(rating);

    try {
      await apiService.submitFeedback(traceId, rating);
      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      setSelectedRating(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Don't show while streaming
  if (isStreaming) return null;

  return (
    <div className="flex items-center gap-1 mt-2">
      {feedbackSubmitted ? (
        <div className="flex items-center gap-1 text-xs text-gray-500">
          <Check size={14} className="text-green-500" />
          <span>Thanks!</span>
        </div>
      ) : (
        <>
          <button
            onClick={() => handleFeedback('like')}
            disabled={isSubmitting}
            className={`p-1.5 rounded-full transition-colors ${
              selectedRating === 'like'
                ? 'bg-green-100 text-green-600'
                : isSubmitting
                ? 'text-gray-300 cursor-not-allowed'
                : 'text-gray-400 hover:bg-green-50 hover:text-green-600'
            }`}
            title="Helpful response"
          >
            <ThumbsUp size={14} />
          </button>
          <button
            onClick={() => handleFeedback('dislike')}
            disabled={isSubmitting}
            className={`p-1.5 rounded-full transition-colors ${
              selectedRating === 'dislike'
                ? 'bg-red-100 text-red-600'
                : isSubmitting
                ? 'text-gray-300 cursor-not-allowed'
                : 'text-gray-400 hover:bg-red-50 hover:text-red-600'
            }`}
            title="Not helpful"
          >
            <ThumbsDown size={14} />
          </button>
        </>
      )}
    </div>
  );
};

export default FeedbackControls;
