"use client";

import { createContext, startTransition, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { PropsWithChildren } from "react";

const READER_FOCUS_STORAGE_KEY = "distribution-training-reader-focus-mode";
const READER_SIDEBAR_WIDTH_STORAGE_KEY = "distribution-training-reader-sidebar-width";

export const READER_SIDEBAR_MIN = 300;
export const READER_SIDEBAR_MAX = 440;
export const READER_SIDEBAR_DEFAULT = 372;

interface ReaderLayoutState {
  focusMode: boolean;
  sidebarWidth: number;
}

interface ReaderLayoutActions {
  setFocusMode: (value: boolean) => void;
  setSidebarWidth: (value: number) => void;
  toggleFocusMode: () => void;
}

const ReaderLayoutStateContext = createContext<ReaderLayoutState | undefined>(undefined);
const ReaderLayoutActionsContext = createContext<ReaderLayoutActions | undefined>(undefined);

function clampSidebarWidth(value: number) {
  return Math.min(READER_SIDEBAR_MAX, Math.max(READER_SIDEBAR_MIN, value));
}

export function ReaderLayoutStateProvider({ children }: PropsWithChildren) {
  const [focusMode, setFocusModeValue] = useState(false);
  const [sidebarWidth, setSidebarWidthValue] = useState(READER_SIDEBAR_DEFAULT);

  useEffect(() => {
    try {
      const savedFocusMode = window.localStorage.getItem(READER_FOCUS_STORAGE_KEY);
      const savedSidebarWidth = window.localStorage.getItem(READER_SIDEBAR_WIDTH_STORAGE_KEY);

      if (savedFocusMode !== null) {
        startTransition(() => {
          setFocusModeValue(savedFocusMode === "1");
        });
      }

      if (savedSidebarWidth !== null) {
        const parsedWidth = Number.parseInt(savedSidebarWidth, 10);

        if (Number.isFinite(parsedWidth)) {
          startTransition(() => {
            setSidebarWidthValue(clampSidebarWidth(parsedWidth));
          });
        }
      }
    } catch {
      // Ignore storage access failures and fall back to the default reading layout.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(READER_FOCUS_STORAGE_KEY, focusMode ? "1" : "0");
      window.localStorage.setItem(READER_SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidth));
    } catch {
      // Ignore storage failures so the layout still works in restrictive environments.
    }
  }, [focusMode, sidebarWidth]);

  const setFocusMode = useCallback((value: boolean) => {
    startTransition(() => {
      setFocusModeValue(value);
    });
  }, []);

  const setSidebarWidth = useCallback((value: number) => {
    startTransition(() => {
      setSidebarWidthValue(clampSidebarWidth(value));
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
      sidebarWidth,
    }),
    [focusMode, sidebarWidth],
  );

  const actionsValue = useMemo(
    () => ({
      setFocusMode,
      setSidebarWidth,
      toggleFocusMode,
    }),
    [setFocusMode, setSidebarWidth, toggleFocusMode],
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
