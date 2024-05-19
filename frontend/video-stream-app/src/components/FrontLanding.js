import React, { useState } from 'react';
import Wave from 'react-wavify'
import "./FrontLanding.css";
import logo from './logo.png'
import SlideShow from './SlideShow';
import VideoStream from './VideoStream';

export default function FrontLanding() {

  return (
    <div style={{backgroundColor: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
      <Wave mask="url(#mask)" fill="#A020F0" >
  <defs>
    <linearGradient id="gradient" gradientTransform="rotate(90)">
      <stop offset="0" stopColor="white" />
      <stop offset="0.5" stopColor="black" />
    </linearGradient>
    <mask id="mask">
      <rect x="0" y="0" width="2000" height="300" fill="url(#gradient)"  />
    </mask>
  </defs>
</Wave>
    <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', height: '150px', marginTop: '50px'}}>
      <img src={logo} style={{width: '800px', height: '175px'}} />
    </div>
    <VideoStream />
  </div>)
}