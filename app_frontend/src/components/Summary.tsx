import React, { useContext, useEffect, useState } from "react";
import { SharedContext } from "../SharedContext";
import Loader from './Loader';

const Summary: React.FC = () => {
  const { summary, setSummary } = useContext(SharedContext)!;
  const { chatbotDisabled } = useContext(SharedContext)!;
  const { loader } = useContext(SharedContext)!;
  const { file } = useContext(SharedContext)!;
  const { selectedDocument } = useContext(SharedContext)!;
  const { deletionInProgress } = useContext(SharedContext)!;
  const [statusMessage, setStatusMessage] = useState<string>('');

  useEffect(() => {
    setSummary('');
  }, [file, setSummary]);

  // Update status message based on chatbot disabled state and loader
  useEffect(() => {
    if (deletionInProgress) {
      setStatusMessage('Deleting document... Please wait.');
    } else if (loader) {
      setStatusMessage('Processing document... Please wait.');
    } else if (chatbotDisabled && !summary) {
      if (selectedDocument) {
        setStatusMessage(`Selected: ${selectedDocument.filename} - Please wait for processing to complete`);
      } else {
        setStatusMessage('Please upload a document to view the summary');
      }
    } else if (chatbotDisabled && summary) {
      setStatusMessage('Document processing in progress...');
    } else {
      setStatusMessage('');
    }
  }, [chatbotDisabled, loader, summary, selectedDocument, deletionInProgress]);

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
  };

  return (
    <>
      <div className="top-row">
        <h3 className="text-center" style={{marginTop:"0.5rem", color:"white"}}>Summary</h3>
        
        {/* Show selected document info */}
        {selectedDocument && (
          <div style={{ 
            backgroundColor: '#1679ab20', 
            padding: '10px', 
            margin: '10px 0', 
            borderRadius: '5px',
            border: '1px solid #1679ab'
          }}>
            <div style={{ color: 'white', fontWeight: 'bold', marginBottom: '5px' }}>
              📄 {selectedDocument.filename}
            </div>
            <div style={{ fontSize: '12px', color: '#bbd0ff', marginBottom: '5px' }}>
              Uploaded: {formatDate(selectedDocument.created_at)}
            </div>
            <div style={{ fontSize: '12px', color: '#bbd0ff' }}>
              Status: {selectedDocument.overall_status.toUpperCase()} | 
              Summary: {selectedDocument.summary_status.toUpperCase()} | 
              RAG: {selectedDocument.rag_status.toUpperCase()}
            </div>
          </div>
        )}
        
        {chatbotDisabled && (
          <div className="col-md-12 summary-upload-message"> 
            {(loader || deletionInProgress) && <Loader/>}
            {statusMessage}
          </div>
        )}
        {!chatbotDisabled && summary && !deletionInProgress && (
          <div className="col-md-12 summary-container">
            <div id="text-summary" style={{ margin: 20, color:"#bbd0ff" }}>
              {summary}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Summary; 