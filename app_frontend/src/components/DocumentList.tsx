import React, { useContext, useEffect, useState } from "react";
import { SharedContext } from "../SharedContext";
import { DocumentInfo } from "../types";
import { apiService } from "../services/api";

const DocumentList: React.FC = () => {
  const { selectedDocument, setSelectedDocument } = useContext(SharedContext)!;
  const { setSummary } = useContext(SharedContext)!;
  const { setDocumentId } = useContext(SharedContext)!;
  const { setTaskId } = useContext(SharedContext)!;
  const { setChatbotDisabled } = useContext(SharedContext)!;
  const { clearDocumentState, setDeletionInProgress, deletionInProgress } = useContext(SharedContext)!;
  
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);

  const fetchDocuments = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getDocuments();
      setDocuments(response.documents);
    } catch (err) {
      setError('Failed to load documents');
      console.error('Error fetching documents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleDocumentSelect = (document: DocumentInfo) => {
    if (deletionInProgress) {
      return;
    }
    
    setSelectedDocument(document);
    setDocumentId(document.document_id);
    setTaskId(document.task_id);
    
    // Set summary if available
    if (document.summary_text) {
      setSummary(document.summary_text);
    }
    
    // Enable chatbot if document is ready
    if (document.overall_status === 'completed' && document.rag_ready) {
      setChatbotDisabled(false);
    } else {
      setChatbotDisabled(true);
    }
  };

  const handleDeleteDocument = async (taskId: string, filename: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent document selection when clicking delete
    
    // Show confirmation dialog
    const confirmed = window.confirm(
      `Are you sure you want to delete "${filename}"?\n\nThis will permanently remove:\n• The document file\n• All embeddings and vector data\n• Processing status and summary\n\nThis action cannot be undone.`
    );
    
    if (!confirmed) {
      return;
    }

    setDeletingTaskId(taskId);
    setDeletionInProgress(true); // Start deletion loader
    
    // If the deleted document was selected, clear content immediately
    if (selectedDocument?.task_id === taskId) {
      clearDocumentState();
    }
    
    try {
      const result = await apiService.deleteDocument(taskId);
      console.log('Document deleted:', result);
      
      // Refresh the document list
      await fetchDocuments();
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      alert(`Failed to delete document: ${errorMessage}`);
      console.error('Error deleting document:', err);
    } finally {
      setDeletingTaskId(null);
      setDeletionInProgress(false); // End deletion loader
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return '#28a745';
      case 'processing':
        return '#ffc107';
      case 'failed':
        return '#dc3545';
      default:
        return '#6c757d';
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
  };

  if (loading) {
    return (
      <div className="document-list">
        <h4 style={{ color: 'white', marginBottom: '10px' }}>Uploaded Documents</h4>
        <div style={{ color: '#bbd0ff', textAlign: 'center' }}>Loading documents...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="document-list">
        <h4 style={{ color: 'white', marginBottom: '10px' }}>Uploaded Documents</h4>
        <div style={{ color: '#ff6b6b', textAlign: 'center' }}>{error}</div>
        <button 
          onClick={fetchDocuments}
          style={{
            backgroundColor: '#1679ab',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            padding: '5px 10px',
            marginTop: '10px',
            cursor: 'pointer'
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="document-list">
      <h4 style={{ color: 'white', marginBottom: '10px' }}>Uploaded Documents</h4>
      {deletionInProgress && (
        <div style={{ 
          color: '#ffc107', 
          textAlign: 'center', 
          marginBottom: '10px',
          fontSize: '12px',
          fontWeight: 'bold'
        }}>
          ⏳ Deletion in progress... Please wait
        </div>
      )}
      {documents.length === 0 ? (
        <div style={{ color: '#bbd0ff', textAlign: 'center' }}>No documents uploaded yet</div>
      ) : (
        <div className="documents-container" style={{ maxHeight: '300px', overflowY: 'auto' }}>
          {documents.map((doc) => (
            <div
              key={doc.document_id}
              className={`document-item ${selectedDocument?.document_id === doc.document_id ? 'selected' : ''}`}
              onClick={() => handleDocumentSelect(doc)}
              style={{
                padding: '10px',
                margin: '5px 0',
                border: selectedDocument?.document_id === doc.document_id ? '2px solid #1679ab' : '1px solid #495057',
                borderRadius: '5px',
                backgroundColor: selectedDocument?.document_id === doc.document_id ? '#1679ab20' : 'transparent',
                cursor: deletionInProgress ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                position: 'relative',
                opacity: deletionInProgress ? 0.6 : 1
              }}
            >
              {/* Delete button */}
              <button
                onClick={(e) => handleDeleteDocument(doc.task_id, doc.filename, e)}
                disabled={deletingTaskId === doc.task_id || deletionInProgress}
                style={{
                  position: 'absolute',
                  top: '5px',
                  right: '5px',
                  backgroundColor: (deletingTaskId === doc.task_id || deletionInProgress) ? '#6c757d' : '#dc3545',
                  color: 'white',
                  border: 'none',
                  borderRadius: '3px',
                  padding: '2px 6px',
                  fontSize: '10px',
                  cursor: (deletingTaskId === doc.task_id || deletionInProgress) ? 'not-allowed' : 'pointer',
                  opacity: (deletingTaskId === doc.task_id || deletionInProgress) ? 0.6 : 1
                }}
                title="Delete document"
              >
                {(deletingTaskId === doc.task_id || deletionInProgress) ? '...' : '×'}
              </button>

              <div style={{ fontWeight: 'bold', color: 'white', marginBottom: '5px', paddingRight: '20px' }}>
                {doc.filename}
              </div>
              <div style={{ fontSize: '12px', color: '#bbd0ff', marginBottom: '5px' }}>
                Uploaded: {formatDate(doc.created_at)}
              </div>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <span style={{ 
                  color: getStatusColor(doc.overall_status),
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {doc.overall_status.toUpperCase()}
                </span>
                {doc.rag_ready && (
                  <span style={{ 
                    color: '#28a745',
                    fontSize: '12px',
                    fontWeight: 'bold'
                  }}>
                    ✓ Ready for Chat
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DocumentList; 