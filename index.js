import * as tf from '@tensorflow/tfjs-node';
// import inkjet from "inkjet"
let model;
// let arrayData
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
    const prediction = await model.predict(tf.reshape(processedImage, [-1, 210, 280, 3]));
    const label = prediction.argMax(1).dataSync()[0];
    const after = Date.now();
    let timeProcess = after - before
    return {
      timeProcess,
      label
    }

  },
  // getLabelFromFile: async function (model, data) {
  //   // console.log('model', model)
  //   let processedImage = await tf.tensor3d(data)
  //   // console.log('processedImage', processedImage)
  //   const prediction = await model.predict(tf.reshape(processedImage, [-1, 210, 280, 3]));
  //   const label = prediction.argMax(1).dataSync()[0];
  //   console.log('label', label)
  // }
}

// do the epic
// let unit8 = async () => {
//   await inkjet.decode(fs.readFileSync('./fl2.jpg'), function (err, decoded) {
//     // decoded: { width: number, height: number, data: Uint8Array }
//     arrayData = arrayToRgbArray(decoded.data)
//   });
// }
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
  // await unit8()
  // CardModel.getLabelFromFile(model, arrayData)
}
main()
// CardModel.x(2, 3)
export default CardModel;