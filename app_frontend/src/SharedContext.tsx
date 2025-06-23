import React, { createContext, useState, ReactNode } from 'react';
import { SharedContextType, DocumentInfo } from './types';

export const SharedContext = createContext<SharedContextType | undefined>(undefined);

interface SharedProviderProps {
  children: ReactNode;
}

export const SharedProvider: React.FC<SharedProviderProps> = ({ children }) => {
  const [summary, setSummary] = useState<string>('');
  const [file, setFile] = useState<File | null>(null);
  const [chatbotDisabled, setChatbotDisabled] = useState<boolean>(true);
  const [loader, setLoader] = useState<boolean>(false);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [selectedDocument, setSelectedDocument] = useState<DocumentInfo | null>(null);
  const [clearChatTrigger, setClearChatTrigger] = useState<number>(0);
  const [deletionInProgress, setDeletionInProgress] = useState<boolean>(false);
  
  // Function to clear all document-related state including chat history
  const clearDocumentState = () => {
    setSummary('');
    setFile(null);
    setChatbotDisabled(true);
    setDocumentId(null);
    setTaskId(null);
    setSelectedDocument(null);
    // Trigger chat history clear by incrementing the trigger
    setClearChatTrigger(prev => prev + 1);
  };
  
  // Function to clear only chat history
  const clearChatHistory = () => {
    setClearChatTrigger(prev => prev + 1);
  };
  
  return (
    <SharedContext.Provider
      value={{
        summary, 
        setSummary,
        file,
        setFile,
        chatbotDisabled,
        setChatbotDisabled,
        loader,
        setLoader,
        documentId,
        setDocumentId,
        taskId,
        setTaskId,
        selectedDocument,
        setSelectedDocument,
        clearChatTrigger,
        clearDocumentState,
        clearChatHistory,
        deletionInProgress,
        setDeletionInProgress
      }}
    >
      {children}
    </SharedContext.Provider>
  );
}; 