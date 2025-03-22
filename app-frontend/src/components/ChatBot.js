import React, {useContext, useState, useEffect, useRef} from "react";
import { SharedContext } from "../SharedContext";
import { IoSend } from "react-icons/io5"

export default function ChatBot(){
    const [question, setQuestion] = useState("");
    const [messages, setMessages] = useState([]);
    const messagesEndRef = useRef(null);
    const { file } = useContext(SharedContext) 
    const { chatbotDisabled } = false;//useContext(SharedContext)
    const { setLoader } = useContext(SharedContext)
    const [ botMessage, setBotMessage ] = useState(null)

    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({behavior: "smooth"})
    }

    useEffect(scrollToBottom, [messages])
    useEffect(() =>{
      setMessages([])
    }, [file])
    // useEffect(() => {
    //   setIsDisabled(setChatbotDisabled)}, [setChatbotDisabled])

    const handleInputChange = (e) => {
      setQuestion(e.target.value)
    }

    const handleKeyPress = (e) => {
      if(e.key === 'Enter'){
        e.preventDefault()
        handleSubmit(e)
      }
    }

    const renderDiv = () =>{
      console.log(chatbotDisabled)
      if(chatbotDisabled){
        return <div className="col-md-12 before-upload-chatarea">Please upload a document to ask questions!</div>
      }
      else{
        return(
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
            )

            }
            <div ref={messagesEndRef}/>
          </div>
        )
      }
    }

    const handleSubmit = async (event) =>{
        setBotMessage("Thinking of an answer")
        event.preventDefault()
        console.log('Inside handleSubmit of the Input.js')
        console.log(file.name)
        console.log(typeof(question))

        setMessages([...messages, {text: question, sender:'user'}])
        setQuestion('');

        var request = {
          "question" : question,
          "filename": file.name,
          "user": "User1"
        }

        try{
          console.log("Before the API call")
            const endpoint = 'http://127.0.0.1:8000/chat'
            const response = await fetch(endpoint, {
              method: "POST",
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify(request)
            })

            console.log(response)

            if(response.ok){
              const data = await response.json()
              console.log(data)
              setBotMessage('')
              setMessages(messages => [...messages, { text: data['answer'], sender: 'bot' }]);
            }else {
              setLoader(false)
              console.error('Server responded with status:', response.status)
              const errorText = await response.text()
              console.error('Error details:', errorText)
            }
        }
        catch(error){
          setBotMessage('')
          console.log(error)
        }        

    }
    return (
        <>
            <div className="bottom-row">
            <h3 className="text-center" style={{marginTop:"0.5rem", color:"white"}}>Ask Me!</h3>
              {/* <div className="col-md-12 messages-container">
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.sender}`}>
                    {message.text}
                  </div>
                ))}
                <div ref={messagesEndRef}/>
              </div> */}
              {renderDiv()}
              <div className="col-md-12" style={{flex: 1}}>
                <form className="input-box" onSubmit={handleSubmit}>
                    <textarea 
                      className="custom-textarea"
                      value={question}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyPress}
                      placeholder="Type your question here..."
                      // disabled={chatbotDisabled}
                    />
                    <button type="submit" className="custom-button"  onClick={handleSubmit}>
                      <IoSend style={{color: '#0081a7', fontSize: '40px'}}/>
                    </button>
                </form>
              </div>
            </div>
        </>
    )
}