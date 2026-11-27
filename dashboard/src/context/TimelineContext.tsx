/**
 * TimelineContext.tsx - Day 0 to 10 forecast timeline scrubber engine
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

interface TimelineContextType {
  currentDay: number; // 0 = initial condition, 1..10 = forecast lead time
  isPlaying: boolean;
  playbackSpeed: number; // 0.5, 1, 2
  setCurrentDay: (day: number) => void;
  togglePlay: () => void;
  setPlaybackSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  resetToDay0: () => void;
}

const TimelineContext = createContext<TimelineContextType | undefined>(undefined);

export const TimelineProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentDay, setCurrentDay] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);

  const stepForward = useCallback(() => {
    setCurrentDay((prev) => (prev >= 10 ? 0 : prev + 1));
  }, []);

  const stepBackward = useCallback(() => {
    setCurrentDay((prev) => (prev <= 0 ? 10 : prev - 1));
  }, []);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const resetToDay0 = useCallback(() => {
    setIsPlaying(false);
    setCurrentDay(0);
  }, []);

  useEffect(() => {
    if (!isPlaying) return;

    const intervalMs = 1200 / playbackSpeed;
    const timer = setInterval(() => {
      setCurrentDay((prev) => {
        if (prev >= 10) {
          return 0; // loop back to Day 0
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed]);

  return (
    <TimelineContext.Provider
      value={{
        currentDay,
        isPlaying,
        playbackSpeed,
        setCurrentDay,
        togglePlay,
        setPlaybackSpeed,
        stepForward,
        stepBackward,
        resetToDay0,
      }}
    >
      {children}
    </TimelineContext.Provider>
  );
};

export function useTimeline(): TimelineContextType {
  const ctx = useContext(TimelineContext);
  if (!ctx) throw new Error('useTimeline must be used within TimelineProvider');
  return ctx;
}
