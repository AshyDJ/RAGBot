import React from 'react'
import { useState } from 'react'
const Test = () => {
    const [prompt,setPrompt]=useState("");
    const [result,setResult]=useState([]);
    const [prompthistory,setPrompthistory]=useState([]);
   const handleOnSubmit= async (e)=>
    {
        e.preventDefault();
        //api call to langflow
        const promptdata={
            prompt:prompt
        };
        const response = await fetch('http://localhost:5000/getprompt', {
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
            setResult(data.msghistory);
            setPrompthistory(data.prompthistory);
            setPrompt("");
          }

    }
  return (
    <>
    <h4>Welcome to your Chat Bot!</h4>
    <form onSubmit={handleOnSubmit}>
    <label>Enter a prompt</label> <input type='text' value={prompt} onChange={(e)=>setPrompt(e.target.value)} required/>
    <button type='submit'>Submit</button>
    </form>


    {prompthistory.map((promptmsg, index) => (
    <div key={`prompt-${index}`}>
    <p><strong>Prompt: </strong> {promptmsg}</p>
    {result[index] && (
      <p><strong>Response: </strong> {result[index]}</p>
    )}
    </div>
    ))}

    </>
  )
}

export default Test
