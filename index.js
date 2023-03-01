console.log("Test JS")
const tf = require('@tensorflow/tfjs');

// import * as tf from '@tensorflow/tfjs';

async function run() {
  const model = await tf.loadLayersModel('https://nfediscover.ml/heart-ml/model.json');
  const prediction = model.predict(tf.tensor2d([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]));
  const classId = prediction.argMax(1).dataSync()[0];
  if (classId==0) {
    console.log("No Disease!");
  }
else {
    console.log("Yes! The person has heart disease!")
}
//   console.log(classId);
}
run();