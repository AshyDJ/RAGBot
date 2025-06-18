import React, { useState, useEffect } from 'react';
import { useNavigate, NavLink } from 'react-router-dom';


const Test = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    fetchSessiondocuments();
  }, []);

  const fetchSessiondocuments = async () => {
    const response = await fetch(`http://127.0.0.1:5000/load_session_documents`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
    });

    if (response.ok) {
      const data = await response.json();
      setDocuments(data.session_documents);
    }
  };

  const Upload_file = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://127.0.0.1:5000/create_session', {
      method: 'POST',
      credentials: 'include',
      body: formData,
    });

    if (response.ok) {
      navigate(`/session/${file.name}`);
    }
  };

  return (
    <>
    

      

      {/* Main Content */}
      <div style={{ marginLeft: '250px', padding: '20px' }}>
        <h2>Welcome to your Chat Bot!</h2>

        <p><b>Create a new Session:</b></p>
        <form onSubmit={Upload_file}>
          <input type="file" onChange={(e) => setFile(e.target.files[0])} />
          <button type="submit">Submit file</button>
        </form>

        <p><b>Sessions:</b></p>
        <ul>
          {documents.map((doc, index) => (
            <li key={index}>
              <NavLink to={`/session/${doc}`} >{doc}</NavLink>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};

export default Test;
