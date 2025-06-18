import React from 'react'
import { useState, useEffect,useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom';
import './Session_chat.css';
import sendIcon from './send.png';
import Side_Navbar from './Side_Navbar'; 

const Session_chat = () => {
    const { file_name } = useParams();
    const [prompt,setPrompt]=useState("");
    const [result,setResult]=useState([]);
    const [chatmemory,setChatmemory]=useState([]);
    const bottomRef = useRef(null);

    useEffect(() => {
            fetchSessionchats()
        }, [file_name]);
    
        const fetchSessionchats=async ()=>{
            const response = await fetch(`http://127.0.0.1:5000/session/${file_name}/load_session_chats`, {
                method: 'GET', 
                headers: {
                  'Content-Type': 'application/json', 
                },
                credentials: 'include',
              });
    
            if(response.ok)
            {
                const data = await response.json(); 
                setChatmemory(data.session_chats)
            }
        }


    useEffect(() => {
    // Scroll to bottom whenever chatmemory changes
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatmemory]);

    const handleOnSubmit= async (e)=>
    {
        e.preventDefault();
        //api call to langflow
        const promptdata={
            prompt:prompt
        };
        const response = await fetch(`http://127.0.0.1:5000/session/${file_name}`, {
            method: 'POST', 
            headers: {
              'Content-Type': 'application/json', 
            },
            credentials: 'include',
            body: JSON.stringify(promptdata), // Convert the data to JSON string
          });

          if (response.ok)
          {
            const data = await response.json(); 
            setResult(data.response);
            setChatmemory(data.chat_memory);
            //setPrompthistory(data.prompthistory);
            setPrompt("");
          }


          

    }


  return (
    <>
      
      <section className="overlay"></section>
  <div className="flex">
    <div className="Conversation">
      {chatmemory.map((chat, index) => (
        <div key={`prompt-${index}`}>
          {index % 2 === 0 ? (
            <p><strong>Prompt:</strong> {chat}</p>
          ) : (
            <p><strong>Response:</strong> {chat}</p>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>

    {/* Fixed prompt area */}
    <div className="Promptdiv">
      <div className="field">
        <form onSubmit={handleOnSubmit}>
          <div className="input-wrapper">
            <input
              className="prompt"
              type="text"
              placeholder="Ask Anything"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              required
            />
            <div className="line"></div>
          </div>
          <button className="submitprompt" type="submit"><img src={sendIcon} /></button>
        </form>
      </div>
    </div>
  </div>
</>

  )
}

export default Session_chat
