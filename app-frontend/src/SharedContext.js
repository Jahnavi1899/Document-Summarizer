import React, {createContext, useState} from 'react';

export const SharedContext = createContext();

export const SharedProvider = ({children}) => {
    const [summary, setSummary] = useState('');
    const [file, setFile] = useState(null);
    const [chatbotDisabled, setChatbotDisabled] = useState(true)
    
    return (
        <SharedContext.Provider
            value={{
                summary, 
                setSummary,
                file,
                setFile,
                chatbotDisabled,
                setChatbotDisabled
            }}
        >
            {children}
        </SharedContext.Provider>
    )
    
}