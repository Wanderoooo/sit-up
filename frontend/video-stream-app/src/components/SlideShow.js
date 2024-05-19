import React from 'react';
import { Carousel } from 'antd';
const contentStyle = {
  height: '160px',
  color: 'blue',
  lineHeight: '300px',
  textAlign: 'center',
  background: 'white',
};
const App = () => (
  <Carousel autoplay>
    <div>
      <h3 style={contentStyle}>1</h3>
    </div>
    <div>
      <h3 style={contentStyle}>2</h3>
    </div>
    <div>
      <h3 style={contentStyle}>3</h3>
    </div>
    <div>
      <h3 style={contentStyle}>4</h3>
    </div>
  </Carousel>
);
export default App;