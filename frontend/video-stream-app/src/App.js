import React, { useEffect } from 'react';
import './App.css';
import FrontLanding from './components/FrontLanding';
import logo from './components/logo.png'
import VideoStream from './components/VideoStream';
import axios from 'axios';

function App() {

  useEffect(() => {
    axios.get('http://localhost:5000/clear_everything')
  } , []);

  return (
    <div>
      <FrontLanding />
      <footer></footer>
    </div>
  );
};

export default App;
