import React, {useContext, useState, useEffect, useRef} from "react";
import { SharedContext } from "../SharedContext";

export default function ChatBot(){
    const [question, setQuestion] = useState("");
    const [messages, setMessages] = useState([]);
    const messagesEndRef = useRef(null);
    const { file } = useContext(SharedContext) 

    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({behavior: "smooth"})
    }

    useEffect(scrollToBottom, [messages])

    const handleInputChange = (e) => {
      setQuestion(e.target.value)
    }

    const handleSubmit = async (event) =>{
        event.preventDefault()
        console.log('Inside handleSubmit of the Input.js')
        console.log(file.name)
        console.log(typeof(question))

        setMessages([...messages, {text: question, sender:'user'}])
        setQuestion('');

        var request = {
          "question" : question,
          "filename": file.name
        }

        try{
          console.log("Before the API call")
            const endpoint = 'http://127.0.0.1:8000/chatbot'
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
              setMessages(messages => [...messages, { text: data.response, sender: 'bot' }]);
            }else {
              console.error('Server responded with status:', response.status)
              const errorText = await response.text()
              console.error('Error details:', errorText)
            }
        }
        catch(error){
          console.log(error)
        }        

    }
    return (
        <>
            <div className="bottom-row">
            <h3 className="text-center">Ask Me!</h3>
              <div className="col-md-12" style={{flex: 6, overflowY: "scroll"}}>
                {messages.map((message, index) => (
                  <div>
                    {message.text}
                  </div>
                ))}
                <div ref={messagesEndRef}/>
              </div>
              <div style={{flex: 2}}>
                <form className="input-box" onSubmit={handleSubmit}>
                    <textarea 
                      className="custom-textarea"
                      value={question}
                      onChange={handleInputChange}
                      placeholder="Ask me here.."
                    />
                    <button type="submit" className="custom-button" onClick={handleSubmit}/>
                </form>
              </div>
                
              </div>
        
        </>
    )
}