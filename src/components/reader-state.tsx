"use client";

import { createContext, startTransition, useCallback, useContext, useMemo, useState } from "react";
import type { PropsWithChildren } from "react";

interface ReaderState {
  collectionLabel: string;
  documentTitle: string;
  activeHeading: string | null;
  progress: number;
}

interface ReaderStateActions {
  clearReaderState: () => void;
  patchReaderState: (value: Partial<ReaderState>) => void;
  replaceReaderState: (value: ReaderState) => void;
}

const ReaderStateContext = createContext<ReaderState | null | undefined>(undefined);
const ReaderStateActionsContext = createContext<ReaderStateActions | undefined>(undefined);

export function ReaderStateProvider({ children }: PropsWithChildren) {
  const [readerState, setReaderState] = useState<ReaderState | null>(null);

  const replaceReaderState = useCallback((value: ReaderState) => {
    startTransition(() => {
      setReaderState(value);
    });
  }, []);

  const patchReaderState = useCallback((value: Partial<ReaderState>) => {
    startTransition(() => {
      setReaderState((currentValue) => (currentValue ? { ...currentValue, ...value } : currentValue));
    });
  }, []);

  const clearReaderState = useCallback(() => {
    startTransition(() => {
      setReaderState(null);
    });
  }, []);

  const contextValue = useMemo(
    () => ({
      clearReaderState,
      patchReaderState,
      replaceReaderState,
    }),
    [clearReaderState, patchReaderState, replaceReaderState],
  );

  return (
    <ReaderStateActionsContext.Provider value={contextValue}>
      <ReaderStateContext.Provider value={readerState}>{children}</ReaderStateContext.Provider>
    </ReaderStateActionsContext.Provider>
  );
}

export function useReaderState() {
  const contextValue = useContext(ReaderStateContext);

  if (contextValue === undefined) {
    throw new Error("useReaderState must be used within ReaderStateProvider.");
  }

  return contextValue;
}

export function useReaderStateActions() {
  const contextValue = useContext(ReaderStateActionsContext);

  if (!contextValue) {
    throw new Error("useReaderStateActions must be used within ReaderStateProvider.");
  }

  return contextValue;
}
