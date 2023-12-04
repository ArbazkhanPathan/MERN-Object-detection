import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from './webcam'; // Replace with the actual path to the 'webcam' file
import RPSDataset from './RPSDataset'; // Replace with the actual path to the 'rps-dataset' file

const RPSGame = () => {
  const videoRef = useRef(null);
  const [rockSamples, setRockSamples] = useState(0);
  const [paperSamples, setPaperSamples] = useState(0);
  const [scissorsSamples, setScissorsSamples] = useState(0);
  const [isPredicting, setIsPredicting] = useState(false);
  const [prediction, setPrediction] = useState('');

  let mobilenet;
  let model;
  const dataset = new RPSDataset();
  const webcam = new Webcam(videoRef.current); 
  async function loadMobilenet() {
    const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = model.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: model.inputs, outputs: layer.output });
  }

  async function train() {
    dataset.ys = null;
    dataset.encodeLabels(3);
    model = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
        tf.layers.dense({ units: 100, activation: 'relu' }),
        tf.layers.dense({ units: 3, activation: 'softmax' }),
      ],
    });
    const optimizer = tf.train.adam(0.0001);
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    let loss = 0;
    await model.fit(dataset.xs, dataset.ys, {
      epochs: 10,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          loss = logs.loss.toFixed(5);
          console.log('LOSS: ' + loss);
        },
      },
    });
  }

  const handleButton = (event) => {
    const buttonId = event.target.id;
    switch (buttonId) {
      case '0':
        setRockSamples((prevRockSamples) => prevRockSamples + 1);
        break;
      case '1':
        setPaperSamples((prevPaperSamples) => prevPaperSamples + 1);
        break;
      case '2':
        setScissorsSamples((prevScissorsSamples) => prevScissorsSamples + 1);
        break;
      default:
        break;
    }

    const label = parseInt(buttonId);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);
  };

  async function predict() {
    while (isPredicting) {
      const predictedClass = tf.tidy(() => {
        const img = webcam.capture();
        const activation = mobilenet.predict(img);
        const predictions = model.predict(activation);
        return predictions.as1D().argMax();
      });
      const classId = (await predictedClass.data())[0];
      let predictionText = '';
      switch (classId) {
        case 0:
          predictionText = 'I see Rock';
          break;
        case 1:
          predictionText = 'I see Paper';
          break;
        case 2:
          predictionText = 'I see Scissors';
          break;
        default:
          break;
      }
      setPrediction(predictionText);

      predictedClass.dispose();
      await tf.nextFrame();
    }
  }

  const doTraining = () => {
    train();
  };

  const startPredicting = () => {
    setIsPredicting(true);
  };

  const stopPredicting = () => {
    setIsPredicting(false);
  };

  useEffect(() => {
    
    const webcam = new Webcam(videoRef.current); // Initialize the Webcam class with the video element
    const init = async () => {
      await webcam.setup(); // Call the 'setup()' function after the video element has been mounted and rendered
      mobilenet = await loadMobilenet();
      tf.tidy(() => mobilenet.predict(webcam.capture()));
    };
    init();
  }, []);

  return (
    <div>
      <div className="video">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          id="wc"
          width="224"
          height="224"
        ></video>
      </div>
      <div className="buttons">
        <button type="button" id="0" onClick={handleButton}>
          Rock
        </button>
        <button type="button" id="1" onClick={handleButton}>
          Paper
        </button>
        <button type="button" id="2" onClick={handleButton}>
          Scissors
        </button>
        <div>Rock Samples: {rockSamples}</div>
        <div>Paper Samples: {paperSamples}</div>
        <div>Scissors Samples: {scissorsSamples}</div>
        <button type="button" onClick={doTraining}>
          Train Network
        </button>
        <div>
          Once training is complete, click 'Start Predicting' to see predictions,
          and 'Stop Predicting' to end
        </div>
        <button type="button" onClick={startPredicting}>
          Start Predicting
        </button>
        <button type="button" onClick={stopPredicting}>
          Stop Predicting
        </button>
        <div id="prediction">{prediction}</div>
      </div>
    </div>
  );
};

export default RPSGame;
