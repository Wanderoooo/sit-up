import { Button, ConfigProvider, Image, Modal, Progress } from "antd";
import React, { useState, useRef } from "react";
import { TinyColor } from "@ctrl/tinycolor";
import { QuestionCircleOutlined } from "@ant-design/icons";
import situp from "./situp.gif";
import slouch from "./slouch.jpg";
import axios from "axios";

const VideoStream = ({api}) => {
  const [sessionStat, setSessionStat] = useState("Start Session"); // Initial countdown value
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [text, setText] = useState("");
  const [sit, setSit] = useState("");
  const [per, setPer] = useState(0);
  const [sec, setSec] = useState(0);
  const colors1 = ["#E0B0FF", "#BF40BF"];
  const [text2, setText2] = useState("");
  const [isTextModalOpen, setIsTextModalOpen] = useState(false);
  const getHoverColors = (colors) =>
    colors.map((color) => new TinyColor(color).lighten(5).toString());
  const getActiveColors = (colors) =>
    colors.map((color) => new TinyColor(color).darken(5).toString());

  const interval = useRef();

  const handleTextCancel = () => {
    setIsTextModalOpen(false);
  };
  const showTextModal = (sessionData) => {
      let ratio = (sessionData.session_total == 0) ? 0 : (sessionData.session_good_posture / sessionData.session_total).toFixed(2);
      setText2(
        `Date: ${sessionData.session_date}, Duration: ${sessionData.session_duration}, session Start time: ${sessionData.session_start_time}, End time: ${sessionData.session_end_time}, Posture score: ${ratio}`
      );
    setIsTextModalOpen(true);
  };

const onSessionClick = (v) => {
    setSessionStat((prev) =>
      prev === "Start Session" ? "End Session" : "Start Session"
    );

    if (v === "Start Session") {
      if (sec === 0) {
        axios.get("http://localhost:5000/start_session");
        interval.current = setInterval(() => {
          setSec((prev) => prev + 1);
        }, 1000);
      }
    } else {
      setSec(0);
      axios.get("http://localhost:5000/end_session");
      axios.get("http://localhost:5000/add_record"); 
      if (interval.current) {
        clearInterval(interval.current);
      }
      axios.get("http://localhost:5000/get_session_data")
      .then((response) => {
        showTextModal(response.data)
      }).catch((error) => { console.error('Error:', error); });

    }
  };
  const handleCancel = () => {
    setIsModalOpen(false);
  };
  const showModal = (stat) => {
    if (stat === "good") {
      setText(
        "Keep your back straight and shoulders relaxed but not slouched. Ensure your feet are flat on the floor and your knees are at a 90-degree angle. Position your screen at eye level to avoid straining your neck."
      );
      setSit(situp);
    } else {
      setText(
        "Calibrate your bad posture by slouching and leaning forward. Ensure your back is rounded and your shoulders are hunched."
      );
      setSit(slouch);
    }
    setIsModalOpen(true);
  };
  const handleOk = () => {
    if (per == 100) {
      setPer(0);
      axios.get("http://localhost:5000/clear_model");
    }
    if (per == 0) {
      console.log("Recording good posture")
      axios.get("http://localhost:5000/record_good_posture");
    }else if (per == 50) {
      console.log("Recording bad posture")
      axios.get("http://localhost:5000/record_bad_posture");
    }

    setIsModalOpen(false);
    let i = 0;

    const interval = setInterval(() => {
      setPer((prev) => {
        if (prev + 5 === 100) {
          console.log("Training model...");
          axios.get("http://localhost:5000/train_model");
          axios.get("http://localhost:5000/clear_training_data");
          clearInterval(interval);
        }
        return prev + 5;
      });
      
      i++;
      if (i === 10) {
        console.log("complete");
        clearInterval(interval);
      }


    }, 1000);

  };

  return (
    <div className="video-wrapper">
      <img
        className="video-stream"
        src="http://localhost:5000/video_feed"
        alt="Video Stream"
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "20px",
        }}
      >
        <ConfigProvider
          theme={{
            components: {
              Button: {
                colorPrimary: `linear-gradient(135deg, ${colors1.join(", ")})`,
                colorPrimaryHover: `linear-gradient(135deg, ${getHoverColors(
                  colors1
                ).join(", ")})`,
                colorPrimaryActive: `linear-gradient(135deg, ${getActiveColors(
                  colors1
                ).join(", ")})`,
                lineWidth: 0,
              },
            },
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "center",
              gap: "10px",
            }}
          >
            <Button
              type="primary"
              size="large"
              onClick={() => showModal("good")}
            >
              Calibrate Good Posture
            </Button>
            <Button
              type="primary"
              size="large"
              onClick={() => showModal("bad")}
            >
              Calibrate Bad Posture
            </Button>
            <Progress
              percent={per}
              type="circle"
              size="small"
              strokeColor="#601ef9"
            />
            <Modal
              title="Calibrate Good Posture"
              open={isModalOpen}
              onOk={handleOk}
              onCancel={handleCancel}
            >
              <Image src={sit} alt="" style={{ width: "100%" }} />
              <p>{text}</p>
            </Modal>
            <Modal
              title="Session Statistics"
              open={isTextModalOpen}
              onOk={handleTextCancel}
              onCancel={handleTextCancel}
            >
              <p>{text2}</p>
            </Modal>
          </div>
        </ConfigProvider>
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "10px",
          }}
        >
          <QuestionCircleOutlined style={{ color: "#601ef9" }} />
          <h7 style={{ color: "#601ef9" }}>
            Calibrate the model to your postures
          </h7>
        </div>
        <ConfigProvider
          theme={{
            components: {
              Button: {
                colorPrimary: `linear-gradient(135deg, ${colors1.join(", ")})`,
                colorPrimaryHover: `linear-gradient(135deg, ${getHoverColors(
                  colors1
                ).join(", ")})`,
                colorPrimaryActive: `linear-gradient(135deg, ${getActiveColors(
                  colors1
                ).join(", ")})`,
                lineWidth: 0,
              },
            },
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "center",
              gap: "10px",
            }}
          >
            <Button
              type="primary"
              size="large"
              onClick={() => onSessionClick(sessionStat)}
            >
              {sessionStat}
            </Button>
            <h1>{`${sec} s`}</h1>
          </div>
        </ConfigProvider>
        <div
          style={{
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: "10px",
          }}
        >
          <QuestionCircleOutlined style={{ color: "#601ef9" }} />
          <h7 style={{ color: "#601ef9" }}>
            Start/End your posture monitoring session
          </h7>
        </div>
      </div>
    </div>
  );
};

export default VideoStream;
