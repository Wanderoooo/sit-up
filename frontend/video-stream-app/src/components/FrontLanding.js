import Wave from 'react-wavify'
import "./FrontLanding.css";
import logo from './logo.png'
import SlideShow from './SlideShow';
import VideoStream from './VideoStream';
import TimeChart from './TimeChart';
import PostureChart from './PostureChart';
import axios from 'axios';
import React, { useMemo } from 'react';
import { notification, Space } from 'antd';
const Context = React.createContext({
  name: 'Default',
});

export default function FrontLanding() {

  let previousPostures = [];
  let postureSensitivity = 0.8;
  let postureNum = 7;
  let send = true;

  const [api, contextHolder] = notification.useNotification();
  const openNotification = (placement) => {
    api.info({
      message: `Bad Posture Detected`,
      description:
        'Please fix your posture!',
      placement,
    });
  };

  setInterval(() => {
    axios.get('http://localhost:5000/predict')
      .then(response => {
        if (response.data.posture !== -1) {
          previousPostures.push(response.data.posture);
          if (previousPostures.length > postureNum) {
            previousPostures.shift();
          }
        }
        if (previousPostures.length === postureNum && 
           previousPostures.reduce((a, b) => a + b, 0) / postureNum > postureSensitivity) {

          send = true;
          console.log('bad posture');
        }
      })
      .catch(error => {
        console.error('Error:', error);
      });
  }, 200);

  setInterval(() => {
    if (send) {
      openNotification('bottomLeft');
      axios.get('http://localhost:5000/send_buzz');
      send = false;
    }
  }, 2000);

  return (
    <div style={{ backgroundColor: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Wave mask="url(#mask)" fill="#A020F0" >
        <defs>
          <linearGradient id="gradient" gradientTransform="rotate(90)">
            <stop offset="0" stopColor="white" />
            <stop offset="0.5" stopColor="black" />
          </linearGradient>
          <mask id="mask">
            <rect x="0" y="0" width="2000" height="300" fill="url(#gradient)" />
          </mask>
        </defs>
      </Wave>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', height: '150px', marginTop: '50px' }}>
        <img src={logo} style={{ width: '800px', height: '175px' }} />
      </div>
      <VideoStream />
      <br />
      <br/>
      <div style={{ display: 'flex', justifyContent: 'space-between'}}>
        <TimeChart style={{height: '150px'}}/>
        <PostureChart style={{height: '150px'}}/>
      </div>
      <br/>
      <br/>
      {contextHolder}
    </div>)
}