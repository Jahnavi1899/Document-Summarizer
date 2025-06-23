export interface SharedContextType {
  summary: string;
  setSummary: (summary: string) => void;
  file: File | null;
  setFile: (file: File | null) => void;
  chatbotDisabled: boolean;
  setChatbotDisabled: (disabled: boolean) => void;
  loader: boolean;
  setLoader: (loading: boolean) => void;
  documentId: string | null;
  setDocumentId: (id: string | null) => void;
  taskId: string | null;
  setTaskId: (id: string | null) => void;
  selectedDocument: DocumentInfo | null;
  setSelectedDocument: (doc: DocumentInfo | null) => void;
  clearChatTrigger: number;
  clearDocumentState: () => void;
  clearChatHistory: () => void;
  deletionInProgress: boolean;
  setDeletionInProgress: (inProgress: boolean) => void;
}

export interface Message {
  text: string;
  sender: 'user' | 'bot';
}

export interface ChatRequest {
  question: string;
  document_id: string;
}

export interface ChatResponse {
  answer: string;
}

export interface UploadResponse {
  message: string;
  task_id: string;
  document_id: string;
  status_endpoint: string;
}

export interface TaskStatusResponse {
  task_id: string;
  document_id: string;
  overall_status: string;
  summary_status: string;
  rag_status: string;
  summary_text: string | null;
  rag_ready: boolean;
  error: string | null;
  filename: string;
  created_at: string;
  last_updated_at: string;
}

export interface DocumentInfo {
  task_id: string;
  document_id: string;
  filename: string;
  overall_status: string;
  summary_status: string;
  rag_status: string;
  summary_text: string | null;
  rag_ready: boolean;
  created_at: string;
  last_updated_at: string;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
} 