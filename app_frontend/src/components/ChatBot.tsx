import React, { useContext, useState, useEffect, useRef } from "react";
import { SharedContext } from "../SharedContext";
import { Message, ChatRequest, ChatResponse } from "../types";
import { apiService } from "../services/api";

const ChatBot: React.FC = () => {
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { file } = useContext(SharedContext)!;
  const { chatbotDisabled } = useContext(SharedContext)!;
  const { setLoader } = useContext(SharedContext)!;
  const { documentId } = useContext(SharedContext)!;
  const { clearChatTrigger, deletionInProgress } = useContext(SharedContext)!;
  const [botMessage, setBotMessage] = useState<string | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState<boolean>(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);
  
  useEffect(() => {
    setMessages([]);
    setChatError(null);
  }, [file]);

  // Clear chat history when clearChatTrigger changes (document deleted)
  useEffect(() => {
    setMessages([]);
    setChatError(null);
    setBotMessage(null);
    setQuestion('');
  }, [clearChatTrigger]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setQuestion(e.target.value);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const retryChat = async () => {
    if (!documentId) {
      alert('No document loaded. Please upload a document first.');
      return;
    }

    setIsRetrying(true);
    setChatError(null);
    
    try {
      // First check if the document is still ready for chat
      const isReady = await apiService.checkDocumentReady(documentId);
      if (!isReady) {
        throw new Error("Document is no longer ready for chat. Please re-upload the document.");
      }

      // Test the chat functionality
      const testSuccess = await apiService.testChat(documentId);
      if (!testSuccess) {
        throw new Error("Chat functionality test failed. Please try again or re-upload the document.");
      }

      setIsRetrying(false);
      setChatError(null);
      console.log("Chat functionality restored successfully");
      
    } catch (error) {
      setIsRetrying(false);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setChatError(`Chat retry failed: ${errorMessage}`);
      console.error("Chat retry failed:", error);
    }
  };

  const renderDiv = () => {
    console.log(chatbotDisabled);
    if (chatbotDisabled || deletionInProgress) {
      const message = deletionInProgress 
        ? "Deleting document... Please wait." 
        : "Please upload a document to ask questions!";
      return <div className="col-md-12 before-upload-chatarea">{message}</div>;
    } else {
      return (
        <div className="col-md-12 messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              {message.text}
            </div>
          ))}
          {botMessage && (
            <div className="message bot thinking">
              {botMessage}
            </div>
          )}
          {chatError && (
            <div className="message bot error">
              <div style={{ color: '#ff6b6b', marginBottom: '10px' }}>
                {chatError}
              </div>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button 
                  onClick={retryChat}
                  disabled={isRetrying}
                  style={{
                    backgroundColor: '#1679ab',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    padding: '8px 16px',
                    cursor: isRetrying ? 'not-allowed' : 'pointer',
                    opacity: isRetrying ? 0.6 : 1
                  }}
                >
                  {isRetrying ? 'Retrying...' : 'Retry Chat'}
                </button>
                <button 
                  onClick={() => {
                    setChatError(null);
                    setMessages([]);
                    // Trigger a page refresh or redirect to upload
                    window.location.reload();
                  }}
                  style={{
                    backgroundColor: '#6c757d',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    padding: '8px 16px',
                    cursor: 'pointer'
                  }}
                >
                  Re-upload Document
                </button>
              </div>
            </div>
          )}
          <div ref={messagesEndRef}/>
        </div>
      );
    }
  };

  const handleSubmit = async (event: React.FormEvent | React.MouseEvent) => {
    if (!documentId) {
      alert('No document loaded. Please upload a document first.');
      return;
    }

    setBotMessage("Thinking of an answer");
    setChatError(null);
    event.preventDefault();
    console.log('Inside handleSubmit of the Input.js');
    if (file) {
      console.log(file.name);
    }
    console.log(typeof(question));

    setMessages([...messages, { text: question, sender: 'user' }]);
    setQuestion('');

    const request: ChatRequest = {
      question: question,
      document_id: documentId
    };

    try {
      console.log("Before the API call");
      const data: ChatResponse = await apiService.sendChatMessage(request);
      console.log(data);
      setBotMessage('');
      setMessages(messages => [...messages, { text: data.answer, sender: 'bot' }]);
    } catch (error) {
      setBotMessage('');
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setChatError(`Chat failed: ${errorMessage}`);
      console.error("Chat error:", error);
    }
  };

  return (
    <>
      <div className="bottom-row">
        <h3 className="text-center" style={{marginTop:"0.5rem", color:"white"}}>Ask Me!</h3>
        {renderDiv()}
        <div className="col-md-12" style={{flex: 1}}>
          <form className="input-box" onSubmit={handleSubmit}>
            <textarea 
              className="custom-textarea"
              value={question}
              onChange={handleInputChange}
              onKeyDown={handleKeyPress}
              placeholder="Type your question here..."
              disabled={!!chatError || deletionInProgress}
            />
            <button 
              type="submit" 
              className="custom-button" 
              onClick={handleSubmit}
              disabled={!!chatError || deletionInProgress}
              style={{ opacity: (chatError || deletionInProgress) ? 0.6 : 1 }}
            >
              <span style={{color: '#0081a7', fontSize: '24px', fontWeight: 'bold'}}>
                →
              </span>
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default ChatBot; 