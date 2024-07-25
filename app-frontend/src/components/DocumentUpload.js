import React, { useContext} from "react";
import { SharedContext } from "../SharedContext";

export default function DocumentUpload(){
    const { file, setFile } = useContext(SharedContext);
    // const [summary, setSummary] = useState('');
    const { setSummary } = useContext(SharedContext);
    const { chatbotDisabled, setChatbotDisabled} = useContext(SharedContext);
    const { setLoader } = useContext(SharedContext)

    const handleFileChange = (event) =>{
        const selectedFile = event.target.files[0]
        setFile(selectedFile)
        console.log('File name:', selectedFile.name)
    }

    const handleUpload = (event) =>{
        event.preventDefault();
        if(file){
            uploadFile(file)
        }
        else{
            alert('Please select a file!')
        }
    }

    const uploadFile = async (file) =>{
        setLoader(true)
        setChatbotDisabled(true)
        console.log("Inside uploadFile method")
        const formData = new FormData()
        formData.append('file', file)
        try{
            console.log("Before the API call")
            const endpoint = 'http://127.0.0.1:8000/upload_file'
            const response = await fetch(endpoint,{
                method:"POST",
                body: formData
            })
           
            console.log("After API call")
            console.log(response)
            if (response.ok){
                console.log("File uploaded successully!")
                setChatbotDisabled(false)
                console.log(chatbotDisabled)
                // console.log(response.json())
                const data = await response.json()
                setSummary(data["summary"])
                setLoader(false)
                // console.log(summary)
                // return data
            }else{
                setLoader(false)
                console.log("File not uploaded")
                alert('Please refresh and try uploading your file again.')
            }
        }
        catch(error){
            setLoader(false)
            console.log(error)
        }
    }

    // const handleClearHistory = () => {

    // }

    return (
        <>
            <div className="upload-left-column">
                <h3 className="text-center" style={{marginTop:"0.5rem", color:"white"}}>Documents</h3>
                <div className="upload-container">
                    <div className="upload-section">
                        <form onSubmit={handleUpload}>  
                            <div style={{marginBottom: "10px"}}>
                                <input type="file" id="pdfUpload" style={{display: "block", width: "100%", color: "#00b4d8"}} onChange={handleFileChange}/>
                            </div>
                            <div>
                                <button type="submit" style={{borderRadius:"10px", backgroundColor:"#1679ab", color: "aliceblue"}}>Upload PDF</button>
                            </div>
                        </form>
                    </div>
                </div>
                <div className="clear-history-section">
                    <button style={{borderRadius:"10px", backgroundColor:"#1679ab", color: "aliceblue"}}>Clear Chat History</button>
                </div>
            </div>
        </>
    )
}



