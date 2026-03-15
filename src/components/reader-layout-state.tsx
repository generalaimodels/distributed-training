"use client";

import { createContext, startTransition, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { PropsWithChildren } from "react";

const READER_FOCUS_STORAGE_KEY = "distribution-training-reader-focus-mode";

interface ReaderLayoutState {
  focusMode: boolean;
}

interface ReaderLayoutActions {
  setFocusMode: (value: boolean) => void;
  toggleFocusMode: () => void;
}

const ReaderLayoutStateContext = createContext<ReaderLayoutState | undefined>(undefined);
const ReaderLayoutActionsContext = createContext<ReaderLayoutActions | undefined>(undefined);

export function ReaderLayoutStateProvider({ children }: PropsWithChildren) {
  const [focusMode, setFocusModeValue] = useState(false);

  useEffect(() => {
    try {
      const savedFocusMode = window.localStorage.getItem(READER_FOCUS_STORAGE_KEY);

      if (savedFocusMode !== null) {
        startTransition(() => {
          setFocusModeValue(savedFocusMode === "1");
        });
      }
    } catch {
      // Ignore storage failures and fall back to the default reading mode.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(READER_FOCUS_STORAGE_KEY, focusMode ? "1" : "0");
    } catch {
      // Ignore storage failures so the layout keeps working in restricted environments.
    }
  }, [focusMode]);

  const setFocusMode = useCallback((value: boolean) => {
    startTransition(() => {
      setFocusModeValue(value);
    });
  }, []);

  const toggleFocusMode = useCallback(() => {
    startTransition(() => {
      setFocusModeValue((currentValue) => !currentValue);
    });
  }, []);

  const stateValue = useMemo(
    () => ({
      focusMode,
    }),
    [focusMode],
  );

  const actionsValue = useMemo(
    () => ({
      setFocusMode,
      toggleFocusMode,
    }),
    [setFocusMode, toggleFocusMode],
  );

  return (
    <ReaderLayoutActionsContext.Provider value={actionsValue}>
      <ReaderLayoutStateContext.Provider value={stateValue}>{children}</ReaderLayoutStateContext.Provider>
    </ReaderLayoutActionsContext.Provider>
  );
}

export function useReaderLayoutState() {
  const contextValue = useContext(ReaderLayoutStateContext);

  if (!contextValue) {
    throw new Error("useReaderLayoutState must be used within ReaderLayoutStateProvider.");
  }

  return contextValue;
}

export function useReaderLayoutActions() {
  const contextValue = useContext(ReaderLayoutActionsContext);

  if (!contextValue) {
    throw new Error("useReaderLayoutActions must be used within ReaderLayoutStateProvider.");
  }

  return contextValue;
}
