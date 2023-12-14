import * as faceapi from 'face-api.js';

import { canvas, faceDetectionNet, faceDetectionOptions, saveFile } from './commons';

const REFERENCE_IMAGE = '../images/bbt1.jpg'
const QUERY_IMAGE = '../images/bbt4.jpg'

async function run() {

  await faceDetectionNet.loadFromDisk('../../weights')
  await faceapi.nets.faceLandmark68Net.loadFromDisk('../../weights')
  await faceapi.nets.faceRecognitionNet.loadFromDisk('../../weights')

  const referenceImage = await canvas.loadImage(REFERENCE_IMAGE)
  const queryImage = await canvas.loadImage(QUERY_IMAGE)

  const resultsRef = await faceapi.detectAllFaces(referenceImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()

  const resultsQuery = await faceapi.detectAllFaces(queryImage, faceDetectionOptions)
    .withFaceLandmarks()
    .withFaceDescriptors()

  const faceMatcher = new faceapi.FaceMatcher(resultsRef)

  const labels = faceMatcher.labeledDescriptors
    .map(ld => ld.label)
  const refDrawBoxes = resultsRef
    .map(res => res.detection.box)
    .map((box, i) => new faceapi.draw.DrawBox(box, { label: labels[i] }))
  const outRef = faceapi.createCanvasFromMedia(referenceImage)
  refDrawBoxes.forEach(drawBox => drawBox.draw(outRef))

  saveFile('referenceImage.jpg', (outRef as any).toBuffer('image/jpeg'))

  // resultsQuery.forEach(fd => {
  //   const bestMatch = faceMatcherRef.findBestMatch(fd.descriptor)
  //   console.log("this is best match thing for query image -  ", bestMatch.toString())
  // })

  // console.log('---------------------');
  // console.log('---------------------');
  // console.log('---------------------');
  // console.log('---------------------');
  
  const queryDrawBoxes = resultsQuery.map(res => {
    const bestMatch = faceMatcher.findBestMatch(res.descriptor)
    // console.log('bestMatch.toString()?', bestMatch.toString());
    // OUPUT:
    // bestMatch.toString()? person 4 (0.53)
    // bestMatch.toString()? unknown (0.86)
    // bestMatch.toString()? person 1 (0.46)
    // bestMatch.toString()? person 2 (0.46)
    // bestMatch.toString()? unknown (0.89)
    // bestMatch.toString()? unknown (0.76)

    return new faceapi.draw.DrawBox(res.detection.box, { label: bestMatch.toString() })
  })
  const outQuery = faceapi.createCanvasFromMedia(queryImage)
  queryDrawBoxes.forEach(drawBox => drawBox.draw(outQuery))
  saveFile('queryImage.jpg', (outQuery as any).toBuffer('image/jpeg'))
  console.log('done, saved results to out/queryImage.jpg')
}

async function mainSequential(){
  let results: any = []

  for (let i = 0; i < 10; i++) {
    var begin=Date.now();
    await run()
    var end= Date.now();
    
    var timeSpent=(end-begin)/1000+"secs";
    results.push(timeSpent)
  }
  console.log('results?', results);
}
async function mainParallel(){
  let results: any = []

  async function runFaceReconition() {
    var begin=Date.now();
    await run()
    var end= Date.now();
    
    var timeSpent=(end-begin)/1000+"secs";
    results.push(timeSpent)
  }
  const promiseArray: any = []
  for (let i = 0; i < 10; i++) {
    promiseArray.push(runFaceReconition())
  }

  await Promise.all(promiseArray)
  
  console.log('results?', results);
}

// Note: Please run accordingly as per benchmark need
mainSequential()
// mainParallel()