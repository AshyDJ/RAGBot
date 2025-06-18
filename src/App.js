import {BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';
//import Test from './components/Test';
import Test from './components/Test_flask';
import Session_chat from './components/Session_chat';
import Side_Navbar from './components/Side_Navbar';
function App() {
  return (
    <>
    <Router>
      <Side_Navbar />
    {/* Overlay to close sidebar when open */}
    <Routes>
    <Route path="/" element={<Test />} />
    <Route path="/session/:file_name" element={<Session_chat />} />
    </Routes>
    </Router>
    </>
    
  );
}

export default App;
