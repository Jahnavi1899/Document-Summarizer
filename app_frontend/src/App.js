import React from 'react';
import './App.css';
import {SharedProvider} from './SharedContext'
import DocumentUpload from './components/DocumentUpload'
import Summary from './components/Summary'
import ChatBot from './components/ChatBot'

// export const NameContext = createContext();
 
// // Create a Context Provider
// const NameContextProvider = ({ children }) => {
//     const [summary, setSummary] = useState(undefined);
 
//     return (
//         <NameContext.Provider value={{ summary, setSummary }}>
//             {children}
//         </NameContext.Provider>
//     );
// };

export default function App(){
  // const [summary, setSummary] = useState('')
  return (
    <>
      <div className="container-fluid">
        <h1 className="text-center" style={{color:"white"}}>Intelligent Document Summarizer</h1>
      </div>
      <div className="container-fluid" style={{ height: "90%" }}>
        <SharedProvider>
          <div className="row" style={{ height: "100%" }}>
              <div className="col-md-3">
                <DocumentUpload/>
              </div>
              <div className="col-md-9">
                <div className="large-section">
                  <Summary/>
                  <ChatBot/>
                </div>
              </div>
          </div>
        </SharedProvider>
      </div>
    </>
  )
  

}