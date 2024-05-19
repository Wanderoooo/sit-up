import { Button, ConfigProvider, Image, Modal, Progress } from 'antd';
import React, { useState, useEffect } from 'react';
import { TinyColor } from '@ctrl/tinycolor';
import { QuestionCircleOutlined } from '@ant-design/icons';
import situp from './situp.gif';
import slouch from './slouch.jpg';

const VideoStream = () => {
  const [sessionStat, setSessionStat] = useState('Start Session'); // Initial countdown value
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [text, setText] = useState('');
  const [sit, setSit] = useState('');
  const [per, setPer] = useState(0);
  const colors1 = ['#E0B0FF', '#BF40BF'];
  const getHoverColors = (colors) =>
    colors.map((color) => new TinyColor(color).lighten(5).toString());
  const getActiveColors = (colors) =>
    colors.map((color) => new TinyColor(color).darken(5).toString());

  const onSessionClick = () => {
    setSessionStat(prev => prev === 'Start Session' ? 'End Session' : 'Start Session');
  }
  const handleCancel = () => {
    setIsModalOpen(false);
  };
  const showModal = (stat) => {
    if (stat === 'good') {
      setText('Keep your back straight and shoulders relaxed but not slouched. Ensure your feet are flat on the floor and your knees are at a 90-degree angle. Position your screen at eye level to avoid straining your neck.')
      setSit(situp)
    } else {
      setText('Calibrate your bad posture by slouching and leaning forward. Ensure your back is rounded and your shoulders are hunched.')
      setSit(slouch)
    }
    setIsModalOpen(true);
  };
  const handleOk = () => {
    setIsModalOpen(false);
    let i = 0;
    const interval = setInterval(() => {
      setPer(prev => prev + 10);
      i++;
      if (i === 5) {
        clearInterval(interval);
      }
    }, 1000);
  };
  
  return (
    <div className="video-wrapper">
      <img className="video-stream" src="http://localhost:5000/video_feed" alt="Video Stream"/>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px'}}>
      <ConfigProvider
      theme={{
        components: {
          Button: {
            colorPrimary: `linear-gradient(135deg, ${colors1.join(', ')})`,
            colorPrimaryHover: `linear-gradient(135deg, ${getHoverColors(colors1).join(', ')})`,
            colorPrimaryActive: `linear-gradient(135deg, ${getActiveColors(colors1).join(', ')})`,
            lineWidth: 0,
          },
        },
      }}
    >
      <div style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: '10px'}}>
      <Button type="primary" size="large" onClick={() => showModal('good')}>
        Calibrate Good Posture
      </Button>
      <Button type="primary" size="large" onClick={() => showModal('bad')}>
        Calibrate Bad Posture
      </Button>
      <Progress percent={per} type="circle" size="small" strokeColor='#601ef9'/>
      <Modal title="Calibrate Good Posture" open={isModalOpen} onOk={handleOk} onCancel={handleCancel}>
      <Image src={sit} alt='' style={{width: '100%'}}/>
      <p>{text}</p>
    </Modal>
      </div>
    </ConfigProvider>
    <div style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: '10px'}}>
    <QuestionCircleOutlined style={{color: '#601ef9'}} />
    <h7 style={{color: '#601ef9'}}>Calibrate the model to your postures</h7>
    </div>
    <ConfigProvider
      theme={{
        components: {
          Button: {
            colorPrimary: `linear-gradient(135deg, ${colors1.join(', ')})`,
            colorPrimaryHover: `linear-gradient(135deg, ${getHoverColors(colors1).join(', ')})`,
            colorPrimaryActive: `linear-gradient(135deg, ${getActiveColors(colors1).join(', ')})`,
            lineWidth: 0,
          },
        },
      }}
    >
      <div style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: '10px'}}>
      <Button type="primary" size="large" onClick={onSessionClick}>
        {sessionStat}
      </Button>
      <h1>1s</h1>
      </div>
    </ConfigProvider>
    <div style={{display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: '10px'}}>
    <QuestionCircleOutlined style={{color: '#601ef9'}} />
    <h7 style={{color: '#601ef9'}}>Start/End your posture monitoring session</h7>
    </div>
    </div>
    </div>
  );
};

export default VideoStream;
