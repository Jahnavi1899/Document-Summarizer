import React, {useContext, useEffect} from "react";
import { SharedContext } from "../SharedContext";
import Loader from './Loader'

export default function Summary(){
    const { summary, setSummary } = useContext(SharedContext);
    const { chatbotDisabled } = useContext(SharedContext)
    const { loader } = useContext(SharedContext)
    const { file } = useContext(SharedContext) 

    useEffect(() =>{
        setSummary('')
      }, [file, setSummary])

    return (
        <>
            <div className="top-row">
            <h3 className="text-center" style={{marginTop:"0.5rem" , color:"white"}}>Summary</h3>
                {chatbotDisabled && <div className="col-md-12 summary-upload-message"> 
                    { loader && <Loader/>}
                    Please upload a document to view the summary</div>
                }
                {
                    !chatbotDisabled && <div className="col-md-12 summary-container">
                    <div id="text-summary" style={{ margin: 20, color:"#bbd0ff" }}>
                        {summary}
                    </div>
                </div>
                }
                
            </div>
        </>
    )
}