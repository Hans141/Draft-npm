import * as tf from '@tensorflow/tfjs';
// console.log('tf.reshape', tf.reshape())
let model;
const initModel = async () => {
  if (!model) model = await tf.loadGraphModel("http://localhost:1231/model.json");
}
let CardModel = {
  getLabelFromCam: async function (model, canvas, video) {
    console.log('model', model)
    let context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, 280, 210);
    let pixel = context.getImageData(0, 0, 280, 210);
    console.log('pixel', pixel.data)
    let data = pixel.data
    let result = arrayToRgbArray(data)
    const before = Date.now();
    let processedImage = await tf.tensor3d(result)
    console.log('processedImage', processedImage)
    const prediction = await model.predict(tf.reshape(processedImage, shape = [-1, 210, 280, 3]));
    const label = prediction.argMax(axis = 1).dataSync()[0];
    const after = Date.now();
    let timeProcess = after - before
    return {
      timeProcess,
      label
    }

  },
  getLabelFromFile: async function (model, fileUrl) {
    console.log('model', model)
  }
}

// do the epic

let arrayToRgbArray = (data) => {
  let input = []
  for (let i = 0; i < 210; i++) {
    input.push([])
    for (let j = 0; j < 280; j++) {
      input[i].push([])
      input[i][j].push(data[(i * 280 + j) * 4])
      input[i][j].push(data[(i * 280 + j) * 4 + 1])
      input[i][j].push(data[(i * 280 + j) * 4 + 2])
    }
  }
  return input
}
let main = async () => {
  await initModel()

}
main()
// CardModel.x(2, 3)
export default CardModel;