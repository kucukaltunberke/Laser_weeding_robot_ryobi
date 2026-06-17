#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point, Point32, Polygon
from std_srvs.srv import Trigger, TriggerResponse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

class YOLOv8TRT:
    def __init__(self, engine_path):
        # Initialize PyCUDA
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.retain_primary_context()
        self.ctx.push()

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size_raw = self.engine.get_binding_shape(binding)
            size = trt.volume(size_raw)
            if size < 0:
                # Handle dynamic shape by forcing absolute size if max_batch=1
                size = abs(size)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': size_raw})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': size_raw})
        
        self.ctx.pop()

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize image with letterbox padding to maintain aspect ratio (matches Ultralytics preprocessing)."""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def infer(self, img_array, is_bgr=True):
        # Push context for multi-threaded ROS environment
        self.ctx.push()
        try:
            # Extract expected model dimensions dynamically (fix operator precedence)
            in_shape = self.inputs[0]['shape']
            (engine_h, engine_w) = (in_shape[2], in_shape[3]) if len(in_shape) >= 4 else (640, 640)
            
            # Pre-processing: Letterbox resize (matches Ultralytics), BGR to RGB, normalize
            img0, ratio, (dw, dh) = self._letterbox(img_array, new_shape=(engine_h, engine_w))
            if is_bgr:
                img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            else:
                img = img0  # already RGB
            img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
            
            # Copy processed image memory to host buffer
            np.copyto(self.inputs[0]['host'], img.ravel())
            
            # Transfer input data to device asynchronously
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Execute model
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer output back to host asynchronously
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            
            # Synchronize the stream
            self.stream.synchronize()
            
            out_shape = self.outputs[0]['shape']
            out = self.outputs[0]['host'].reshape(out_shape)
            
            # DEBUG: Log output shape and value ranges to diagnose detection issues
            rospy.logwarn("DEBUG TRT output shape: {}, dtype: {}".format(out.shape, out.dtype))
            squeezed = np.squeeze(out)
            rospy.logwarn("DEBUG squeezed shape: {}".format(squeezed.shape))
            rospy.logwarn("DEBUG output min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
                float(np.min(squeezed)), float(np.max(squeezed)), float(np.mean(squeezed))))
            # Log value ranges for first few rows/columns to understand layout
            if len(squeezed.shape) == 2:
                dim0, dim1 = squeezed.shape
                # Check what's at index 4 (where we expect confidence)
                if dim0 < dim1:  # (5, 8400) layout
                    rospy.logwarn("DEBUG row[4] (expected conf) min: {:.4f}, max: {:.4f}".format(
                        float(np.min(squeezed[4, :])), float(np.max(squeezed[4, :]))))
                    for i in range(min(dim0, 10)):
                        rospy.logwarn("DEBUG row[{}] range: [{:.4f}, {:.4f}]".format(
                            i, float(np.min(squeezed[i, :])), float(np.max(squeezed[i, :]))))
                else:  # (8400, 5) layout
                    rospy.logwarn("DEBUG col[4] (expected conf) min: {:.4f}, max: {:.4f}".format(
                        float(np.min(squeezed[:, 4])), float(np.max(squeezed[:, 4]))))
            
            # Post-processing
            # YOLOv8 target output usually (1, 4+num_classes, N) => (4+1, 8400) for 1 class
            out = squeezed
            
            # Ensure it is in expected shape of (8400, 5) if it transposed during export
            if out.shape[0] < out.shape[1]:
                out = out.transpose(1, 0) # shape (8400, 5)
            
            boxes = []
            scores = []
            
            orig_h, orig_w = img_array.shape[:2]
            
            # Filter bounding boxes from output array
            num_classes = out.shape[1] - 4  # first 4 are box coords, rest are class scores
            rospy.logwarn("DEBUG num_classes detected: {}".format(num_classes))
            for row in out:
                class_scores = row[4:]  # all class confidences
                confidence = float(np.max(class_scores))
                class_id = int(np.argmax(class_scores))
                if confidence > 0.15:  # Lowered to match working script's conf=0.15
                    xc, yc, w, h = row[0], row[1], row[2], row[3]
                    
                    # Undo letterbox padding and scale back to original image coordinates
                    left = int(((xc - w / 2) - dw) / ratio)
                    top = int(((yc - h / 2) - dh) / ratio)
                    width = int(w / ratio)
                    height = int(h / ratio)
                    
                    # Clamp to image boundaries
                    left = max(0, left)
                    top = max(0, top)
                    width = min(width, orig_w - left)
                    height = min(height, orig_h - top)
                    
                    boxes.append([left, top, width, height])
                    scores.append(float(confidence))
                    
            # Apply OpenCV NMS Filtering
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.23, 0.45)
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    results.append({
                        "left": box[0],
                        "top": box[1],
                        "right": box[0] + box[2],
                        "bottom": box[1] + box[3],
                        "conf": scores[i]
                    })
            return results
        finally:
            self.ctx.pop()


class WeedDetector:
    def __init__(self):
        # 1. Initialize the ROS node
        rospy.init_node('weed_detector_yolo', anonymous=True)

        # 2. Load your custom trained model directly from TensorRT PyCUDA wrapper
        import rospkg, os
        model_path = os.path.join(rospkg.RosPack().get_path('image_processing'), 'weight', 'new_weed_detector.engine')
        self.model = YOLOv8TRT(model_path)

        # 4. Use message_filters to synchronize RGB and Depth images
        import message_filters
        # Set queue_size=1 so ROS drops old frames instead of building a backlog
        self.rgb_sub = message_filters.Subscriber("/zedm/zed_node/left/image_rect_color", Image, queue_size=1, buff_size=2**24)
        self.depth_sub = message_filters.Subscriber("/zedm/zed_node/depth/depth_registered", Image, queue_size=1, buff_size=2**24)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        import sys, os, rospkg
        script_path = os.path.join(rospkg.RosPack().get_path('image_processing'), 'scripts')
        sys.path.append(script_path)
        from weed_coordinate_solver import WeedCoordinateSolver
        self.coordinate_solver = WeedCoordinateSolver()

        self.image_pub = rospy.Publisher("/yolo/annotated_image/compressed", CompressedImage, queue_size=1)
        self.weed_list_pub = rospy.Publisher("/weed_list", Polygon, queue_size=10)

        self.latest_rgb_data = None
        self.latest_depth_data = None
        self.process_srv = rospy.Service('/process_latest_image', Trigger, self.process_image_srv)

    def image_callback(self, rgb_data, depth_data):
        self.latest_rgb_data = rgb_data
        self.latest_depth_data = depth_data

    def process_image_srv(self, req):
        if self.latest_rgb_data is None or self.latest_depth_data is None:
            rospy.logwarn("Process requested, but no image received yet.")
            return TriggerResponse(success=False, message="No image received yet")

        rgb_data = self.latest_rgb_data
        depth_data = self.latest_depth_data

        try:
            # 5. Convert the incoming ROS Image message directly into an OpenCV image (NumPy array)
            channels = 4 if 'bgra' in rgb_data.encoding or 'rgba' in rgb_data.encoding else 3
            cv_image = np.ndarray(shape=(rgb_data.height, rgb_data.width, channels),
                                  dtype=np.uint8, buffer=rgb_data.data,
                                  strides=(rgb_data.step, channels, 1))
            
            if channels == 4:
                cv_image = cv_image[:, :, :3]  # Drop the alpha channel for YOLO

            # FORCE the array to be C-contiguous
            cv_image = np.ascontiguousarray(cv_image)

            # OpenCV handles depth images as 32-bit floats
            cv_depth = np.ndarray(shape=(depth_data.height, depth_data.width),
                                  dtype=np.float32, buffer=depth_data.data,
                                  strides=(depth_data.step, 4))
        except Exception as e:
            rospy.logerr("Image Conversion Error: {}".format(e))
            return TriggerResponse(success=False, message="Image Conversion Error: {}".format(e))

        # Determine if cv_image is in BGR or RGB order based on the camera encoding
        is_bgr = 'bgr' in rgb_data.encoding.lower()

        # 6. Run TensorRT inference natively on GPU execution context
        import time
        start_time = time.time()
        results = self.model.infer(cv_image, is_bgr=is_bgr)
        rospy.loginfo("Inference processed in {:.2f} seconds".format(time.time() - start_time))

        import cv2
        annotated_frame = cv_image.copy()
        weed_polygon = Polygon()

        # 7. Extract the pixel coordinates for your inverse kinematics
        rgb_h, rgb_w = cv_image.shape[:2]
        depth_h, depth_w = cv_depth.shape[:2]
        scale_x = depth_w / float(rgb_w)
        scale_y = depth_h / float(rgb_h)

        for res in results:
            x1, y1 = res["left"], res["top"]
            x2, y2 = res["right"], res["bottom"]

            # Calculate the exact center pixel (u, v) in RGB image space
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)
            
            # Scale to depth image space for depth lookup
            u_depth = max(0, min(int(u * scale_x), depth_w - 1))
            v_depth = max(0, min(int(v * scale_y), depth_h - 1))
            
            confidence = res["conf"]
            
            # Fetch metric distance from depth camera (using depth-space coords)
            depth_val = cv_depth[v_depth, u_depth]
            coords_3d = self.coordinate_solver.get_3d_coordinate(u, v, depth_val)
            
            coord_label = "No Depth"
            if coords_3d:
                x_g, y_g, z_g = coords_3d
                rospy.loginfo("Weed at pixel ({}, {}) | Confidence: {:.2f} | 3D Ground: X={:.2f}m, Y={:.2f}m, Z={:.2f}m".format(u, v, confidence, x_g, y_g, z_g))
                
                p = Point32()
                p.x = x_g
                p.y = y_g
                p.z = z_g
                weed_polygon.points.append(p)
                coord_label = "({:.2f}, {:.2f}, {:.2f})m".format(x_g, y_g, z_g)
            else:
                rospy.loginfo("Weed at pixel ({}, {}) | Confidence: {:.2f} | 3D: Invalid depth".format(u, v, confidence))
            
            # Draw a green bounding box (BGR: 0, 255, 0)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            
            # Draw a small red dot at the center coordinates (RGB-space, not clamped)
            cv2.circle(annotated_frame, (u, v), 5, (0, 0, 255), -1)
            
            # Write the confidence and 3D coordinates near the box
            label = "Weed {:.2f} {}".format(confidence, coord_label)
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert the manually annotated image to a compressed JPEG to avoid shape mismatches
        try:
            msg = CompressedImage()
            msg.header = rgb_data.header
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', annotated_frame)[1]).tobytes()
            self.image_pub.publish(msg)
        except Exception as e:
            rospy.logerr("Image Publish Error: {}".format(e))
                
        self.weed_list_pub.publish(weed_polygon)
        return TriggerResponse(success=True, message="Image processed successfully")

if __name__ == '__main__':
    try:
        detector = WeedDetector()
        rospy.loginfo("YOLOv8 TensorRT Weed Detection Node Started. Waiting for images...")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
