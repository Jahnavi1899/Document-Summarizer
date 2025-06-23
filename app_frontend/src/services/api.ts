import { UploadResponse, TaskStatusResponse, ChatRequest, ChatResponse, DocumentsResponse } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const apiService = {
  // Upload file and get task information
  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/upload_file`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Get list of uploaded documents
  async getDocuments(): Promise<DocumentsResponse> {
    const response = await fetch(`${API_BASE_URL}/documents`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Get task status
  async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Check if document is ready for chat
  async checkDocumentReady(documentId: string): Promise<boolean> {
    try {
      // Try to find the document in task status collection
      const response = await fetch(`${API_BASE_URL}/status/${documentId}`);
      if (response.ok) {
        const status = await response.json();
        return status.rag_ready === true && status.overall_status === 'completed';
      }
      return false;
    } catch (error) {
      console.error("Error checking document readiness:", error);
      return false;
    }
  },

  // Send chat message
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chat failed: ${errorText}`);
    }
    
    return response.json();
  },

  // Test chat functionality
  async testChat(documentId: string): Promise<boolean> {
    try {
      const testRequest: ChatRequest = {
        question: "Hello, can you confirm you can access the document?",
        document_id: documentId
      };
      
      await this.sendChatMessage(testRequest);
      return true;
    } catch (error) {
      console.error("Chat test failed:", error);
      return false;
    }
  },

  // Delete document and all associated data
  async deleteDocument(taskId: string): Promise<{
    message: string;
    deleted_task_id: string;
    deleted_document_id: string;
    deleted_filename: string;
    embeddings_removed: number;
  }> {
    const response = await fetch(`${API_BASE_URL}/documents/${taskId}`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Delete failed: ${errorText}`);
    }
    
    return response.json();
  },

  // Delete all documents and their associated data
  async deleteAllDocuments(): Promise<{
    message: string;
    documents_removed: number;
    embeddings_removed: number;
    files_removed: number;
  }> {
    const response = await fetch(`${API_BASE_URL}/documents`, {
      method: 'DELETE'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Delete all failed: ${errorText}`);
    }
    
    return response.json();
  }
}; 