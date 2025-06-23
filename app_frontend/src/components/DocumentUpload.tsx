import React, { useContext, useEffect, useState } from "react";
import { SharedContext } from "../SharedContext";
import { apiService } from "../services/api";
import DocumentList from "./DocumentList";

const DocumentUpload: React.FC = () => {
  const { file, setFile } = useContext(SharedContext)!;
  const { summary, setSummary } = useContext(SharedContext)!;
  const { setChatbotDisabled } = useContext(SharedContext)!;
  const { setLoader } = useContext(SharedContext)!;
  const { setDocumentId } = useContext(SharedContext)!;
  const { setTaskId } = useContext(SharedContext)!;
  const { selectedDocument, setSelectedDocument } = useContext(SharedContext)!;
  const { clearDocumentState, clearChatHistory, setDeletionInProgress, deletionInProgress } = useContext(SharedContext)!;
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [overallStatus, setOverallStatus] = useState<string>('');
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      console.log('File name:', selectedFile.name);
    }
  };

  const handleUpload = (event: React.FormEvent) => {
    event.preventDefault();
    if (file) {
      uploadFile(file);
    } else {
      alert('Please select a file!');
    }
  };

  const uploadFile = async (file: File) => {
    setLoader(true);
    setProcessingError(null);
    setSummary(''); // Clear any previous summary
    setOverallStatus(''); // Clear previous status
    setSelectedDocument(null); // Clear selected document
    setChatbotDisabled(true); // Ensure chatbot is disabled on new upload
    console.log("Inside uploadFile method");
    
    try {
      console.log("Before the API call");
      const uploadResponse = await apiService.uploadFile(file);
      console.log("File upload initiated:", uploadResponse);
      setTaskId(uploadResponse.task_id);
      setDocumentId(uploadResponse.document_id);
      pollTaskStatus(uploadResponse.task_id);
      
      // Trigger refresh of document list
      setRefreshTrigger(prev => prev + 1);
    } catch (error) {
      setLoader(false);
      setProcessingError('Upload failed. Please try again.');
      console.error("Upload error:", error);
      alert('Upload failed. Please try again.');
    }
  };

  const pollTaskStatus = async (taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await apiService.getTaskStatus(taskId);
        console.log("Task status:", status);
        
        // Update summary if available
        if (status.summary_text) {
          setSummary(status.summary_text);
        }
        
        // Update overall status
        setOverallStatus(status.overall_status);
        
        // Check if processing is complete (success or failure)
        if (status.overall_status === 'failed') {
          clearInterval(pollInterval);
          setLoader(false);
          setProcessingError(status.error || 'Processing failed');
          console.error("Processing failed:", status.error);
          alert(`Processing failed: ${status.error || 'Unknown error'}`);
        } else if (status.overall_status === 'completed') {
          // Continue polling until summary is available
          if (status.summary_text && status.summary_text.trim() !== "") {
            clearInterval(pollInterval);
            setLoader(false);
            console.log("Both overall status completed and summary available");
            
            // Refresh document list after successful completion
            setRefreshTrigger(prev => prev + 1);
          } else {
            console.log("Overall status completed but waiting for summary...");
          }
        }
        
        // Check for individual task errors
        if (status.error) {
          clearInterval(pollInterval);
          setLoader(false);
          setProcessingError(status.error);
          console.error("Processing error:", status.error);
          alert(`Processing error: ${status.error}`);
        }
      } catch (error) {
        clearInterval(pollInterval);
        setLoader(false);
        setProcessingError('Failed to check processing status');
        console.error("Status check error:", error);
        alert('Failed to check processing status. Please refresh and try again.');
      }
    }, 2000);
    
    setTimeout(() => {
      clearInterval(pollInterval);
      setLoader(false);
      setProcessingError('Processing timeout');
      alert('Processing is taking longer than expected. Please check back later.');
    }, 300000);
  };

  // Enable chatbot only when overall processing is completed successfully
  useEffect(() => {
    console.log(`Overall status: ${overallStatus}, Summary available: ${!!summary}, Error: ${!!processingError}`);
    
    if (overallStatus === 'completed' && summary && summary.trim() !== "" && !processingError) {
      setChatbotDisabled(false);
      console.log("Document processing completed successfully - chatbot enabled");
    } else {
      setChatbotDisabled(true);
      if (overallStatus === 'completed' && (!summary || summary.trim() === "")) {
        console.log("Overall status is completed but summary is not available yet");
      }
    }
  }, [overallStatus, summary, processingError, setChatbotDisabled]);

  return (
    <>
      <div className="upload-left-column">
        <h3 className="text-center" style={{marginTop:"0.5rem", color:"white"}}>Documents</h3>
        
        <div className="upload-container">
          <div className="upload-section">
            <form onSubmit={handleUpload}>  
              <div style={{marginBottom: "10px"}}>
                <input 
                  type="file" 
                  id="pdfUpload" 
                  style={{
                    display: "block", 
                    width: "100%", 
                    color: "#00b4d8",
                    opacity: deletionInProgress ? 0.6 : 1,
                    cursor: deletionInProgress ? 'not-allowed' : 'pointer'
                  }} 
                  onChange={handleFileChange}
                  accept=".pdf"
                  disabled={deletionInProgress}
                />
              </div>
              <div>
                <button 
                  type="submit" 
                  style={{
                    borderRadius:"10px", 
                    backgroundColor: deletionInProgress ? "#6c757d" : "#1679ab", 
                    color: "aliceblue",
                    opacity: deletionInProgress ? 0.6 : 1,
                    cursor: deletionInProgress ? 'not-allowed' : 'pointer'
                  }}
                  disabled={deletionInProgress}
                >
                  Upload PDF
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Document List */}
        <DocumentList key={refreshTrigger} />
        
        <div className="clear-history-section">
          <button 
            onClick={async () => {
              const confirmed = window.confirm(
                "Are you sure you want to delete ALL documents?\n\nThis will permanently remove:\n• All document files\n• All embeddings and vector data\n• All processing status and summaries\n\nThis action cannot be undone."
              );
              
              if (!confirmed) {
                return;
              }

              setDeletionInProgress(true); // Start deletion loader
              
              // Clear all document state immediately
              clearDocumentState();

              try {
                const result = await apiService.deleteAllDocuments();
                console.log('All documents deleted:', result);
                
                // Refresh document list
                setRefreshTrigger(prev => prev + 1);
                
                alert(`Successfully deleted ${result.documents_removed} documents, ${result.embeddings_removed} embeddings, and ${result.files_removed} files.`);
              } catch (error) {
                const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                alert(`Failed to delete all documents: ${errorMessage}`);
                console.error('Error deleting all documents:', error);
              } finally {
                setDeletionInProgress(false); // End deletion loader
              }
            }}
            style={{
              borderRadius:"10px", 
              backgroundColor: deletionInProgress ? "#6c757d" : "#dc3545", 
              color: "white",
              border: 'none',
              padding: '8px 16px',
              cursor: deletionInProgress ? 'not-allowed' : 'pointer',
              width: '100%',
              marginBottom: '10px',
              opacity: deletionInProgress ? 0.6 : 1
            }}
            disabled={deletionInProgress}
          >
            {deletionInProgress ? 'Deleting...' : 'Clear All Documents'}
          </button>
          <button 
            onClick={() => {
              // Clear only chat history
              clearChatHistory();
            }}
            style={{
              borderRadius:"10px", 
              backgroundColor: deletionInProgress ? "#6c757d" : "#1679ab", 
              color: "aliceblue",
              opacity: deletionInProgress ? 0.6 : 1,
              cursor: deletionInProgress ? 'not-allowed' : 'pointer'
            }}
            disabled={deletionInProgress}
          >
            Clear Chat History
          </button>
        </div>
      </div>
    </>
  );
};

export default DocumentUpload; 