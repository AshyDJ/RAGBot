import React, { useState,useEffect } from 'react';
import { useNavigate, NavLink } from 'react-router-dom';
import './Side_Navbar.css'; // Make sure this CSS is imported

const Side_Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleSidebar = () => setIsOpen(!isOpen);
  const closeSidebar = () => setIsOpen(false);
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









  return (
    <>
    <link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet" />

      <nav className={isOpen ? 'open' : ''}>
        <div className="logo">
          <i className="bx bx-menu menu-icon" onClick={toggleSidebar}></i>
          <span className="logo-name">
            <NavLink to={`/`} className='links'>RAG </NavLink>
            </span>
        </div>

        <div className="sidebar">
          <div className="logo">
            <i className="bx bx-menu menu-icon" onClick={toggleSidebar}></i>
            <span className="logo-name">
                <NavLink to={`/`} className='links'>RAG </NavLink>
                </span>
          </div>
          <div className="sidebar-content">


            <ul className="lists">
                {documents.map((doc, index) => (
                    <li key={index} className="list">
                    <NavLink to={`/session/${doc}`} className="nav-link">
                    <span className="link">{doc}</span>
                    </NavLink>
                    </li>
                    ))}
            </ul>

           
            
          </div>
        </div>
      </nav>

      <section className="overlay" onClick={closeSidebar}></section>
    </>
  );
};
 
export default Side_Navbar;
/*
            <ul className="lists">
              <li className="list">
                <a href="#" className="nav-link">
                  <i className="bx bx-folder-open icon"></i>
                  <span className="link">Files</span>
                </a>
              </li>
            </ul>
                */