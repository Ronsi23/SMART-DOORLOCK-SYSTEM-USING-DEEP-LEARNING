import cv2
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import face_recognition
import lgpio

# GPIO Control via sysfs (paling stabil)
class GPIOController:
    def __init__(self, pin):
        self.pin = pin
        try:
            # Buka GPIO chip
            self.gpio_chip = lgpio.gpiochip_open(0)
            # Set pin sebagai output
            lgpio.gpio_claim_output(self.gpio_chip, self.pin)
            self.off()
            print(f"✅ GPIO {self.pin} setup berhasil via lgpio")
        except Exception as e:
            print(f"❌ Error inisialisasi lgpio: {e}")
            self.gpio_chip = None

    def on(self):
        try:
            if self.gpio_chip is not None:
                lgpio.gpio_write(self.gpio_chip, self.pin, 1)
                print(f"✅ GPIO {self.pin} ON")
        except Exception as e:
            print(f"⚠️ Error GPIO ON: {e}")
    
    def off(self):
        try:
            if self.gpio_chip is not None:
                lgpio.gpio_write(self.gpio_chip, self.pin, 0)
                print(f"✅ GPIO {self.pin} OFF")
        except Exception as e:
            print(f"⚠️ Error GPIO OFF: {e}")
    
    def cleanup(self):
        try:
            if self.gpio_chip is not None:
                self.off()
                lgpio.gpio_free(self.gpio_chip, self.pin)
                lgpio.gpiochip_close(self.gpio_chip)
                print(f"✅ GPIO {self.pin} cleanup berhasil")
        except Exception as e:
            print(f"⚠️ Warning cleanup GPIO: {e}")

# Coba import gaze_tracking, jika gagal gunakan OpenCV
try:
    from gaze_tracking import GazeTracking
    GAZE_TRACKING_AVAILABLE = True
    print("✅ Library gaze_tracking berhasil dimuat")
except ImportError:
    GAZE_TRACKING_AVAILABLE = False
    print("⚠️ Library gaze_tracking tidak tersedia, menggunakan OpenCV fallback")

class IntegratedVerificationSystem:
    def __init__(self):
        print("Memulai sistem verifikasi terintegrasi...")
        
        # Load model
        try:
            self.model, self.input_details, self.output_details = self.load_tflite_model("/home/admin/TA/model_final2.tflite")
            print("✅ Model berhasil dimuat")
        except Exception as e:
            raise Exception(f"Gagal memuat model: {str(e)}")
        
        # Load label dari dataset test
        self.label_names = self.load_label_names("/home/admin/TA/split/test")
        print(f"✅ Dataset dimuat: {list(self.label_names.values())}")
        
        # Inisialisasi gaze tracking jika tersedia
        if GAZE_TRACKING_AVAILABLE:
            try:
                self.gaze = GazeTracking()
                self.use_gaze_tracking = True
                print("✅ GazeTracking diinisialisasi")
            except Exception as e:
                print(f"⚠️ Gagal inisialisasi GazeTracking: {e}")
                self.use_gaze_tracking = False
        else:
            self.use_gaze_tracking = False
            
        # Load OpenCV classifiers sebagai fallback
        if not self.use_gaze_tracking:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                print("✅ OpenCV classifiers dimuat sebagai fallback")
            except Exception as e:
                print(f"⚠️ Error loading OpenCV classifiers: {e}")
        
        self.last_face_result = (None, 0, None)

        # Daftar perintah
        self.commands = ["LIHAT KIRI", "LIHAT KANAN", "LIHAT TENGAH", "KEDIP"]
        
        # Inisialisasi kamera
        self.cap = self.initialize_camera()

        # Inisialisasi relay dengan sysfs
        try:
            self.RELAY_PIN = 18
            self.gpio_controller = GPIOController(self.RELAY_PIN)
            print("✅ Relay GPIO 18 diinisialisasi dengan sysfs")
            print("✅ Wiring: Raspi 5V→Relay VCC, Raspi GND→Relay GND, GPIO18→Relay IN")
            print("✅ Relay: COM→PSU V+, NO→Solenoid+, PSU V-→Solenoid-")
            
            # Test relay untuk memastikan berfungsi
            print("🔧 Testing relay...")
            self.gpio_controller.on()
            time.sleep(0.2)
            self.gpio_controller.off()
            print("✅ Test relay berhasil")
            
        except Exception as e:
            print(f"⚠️ Warning: Gagal inisialisasi relay: {e}")
            self.gpio_controller = None

        # Variabel untuk kontrol loop
        self.system_running = True
        self.reset_verification_state()

    def reset_verification_state(self):
        """Reset state untuk memulai verifikasi baru"""
        # State management untuk verifikasi bertahap
        self.current_stage = "DETECTING_FACE"
        self.detected_face_name = None
        self.current_command = None
        self.command_start_time = None
        
        # Counter untuk stabilitas deteksi
        self.face_stable_count = 0
        self.action_stable_count = 0
        
        # Untuk OpenCV fallback
        self.eye_closed_frames = 0
        self.eye_open_frames = 0
        self.blink_detected = False
        
        # Status verifikasi
        self.verification_complete = False
        self.verification_success = False
        
        # Timer untuk timeout per sesi
        self.start_time = time.time()
        self.timeout_seconds = 50

    def load_tflite_model(self, model_path):
        """Load TensorFlow Lite model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
            
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    
    def load_label_names(self, dataset_path):
        """Memuat nama label dari folder dataset test"""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_path}")
        
        labels = sorted([
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ])
        
        if not labels:
            raise ValueError("Tidak ada data wajah yang terdaftar")
        
        return {i: label for i, label in enumerate(labels)}

    def initialize_camera(self):
        """Inisialisasi kamera dengan multiple fallback"""
        for i in range(5):  # Coba lebih banyak camera index
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Test baca frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Resolusi yang lebih stabil
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        print(f"✅ Kamera berhasil diakses (ID: {i})")
                        return cap
                cap.release()
            except Exception as e:
                print(f"⚠️ Camera {i} error: {e}")
                continue
        raise Exception("❌ Tidak dapat mengakses kamera")

    def detect_action_gaze_tracking(self, frame):
        """Deteksi aksi menggunakan gaze_tracking library"""
        try:
            self.gaze.refresh(frame)
            
            # Gambar tanda + pada pupil mata
            annotated_frame = self.gaze.annotated_frame()
            if annotated_frame is not None:
                # Copy hasil annotated frame ke frame asli
                frame[:] = annotated_frame[:]
            
            action = ""
            if self.gaze.is_blinking():
                action = "KEDIP"
            elif self.gaze.is_right():
                action = "LIHAT KANAN"
            elif self.gaze.is_left():
                action = "LIHAT KIRI"
            elif self.gaze.is_center():
                action = "LIHAT TENGAH"
                
            return action
        except Exception as e:
            print(f"⚠️ Gaze tracking error: {e}")
            return ""

    def detect_action_opencv_fallback(self, frame):
        """Deteksi aksi menggunakan OpenCV sebagai fallback"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) == 0:
                return ""
            
            # Ambil wajah pertama
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Deteksi mata untuk kedip dan gambar tanda +
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(15, 15))
            
            # Gambar tanda + pada mata yang terdeteksi (simulasi pupil tracking)
            for (ex, ey, ew, eh) in eyes:
                # Konversi koordinat mata ke frame asli
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                
                # Gambar tanda + pada center mata
                cross_size = 10
                cv2.line(frame, 
                        (eye_center_x - cross_size, eye_center_y), 
                        (eye_center_x + cross_size, eye_center_y), 
                        (0, 255, 0), 2)
                cv2.line(frame, 
                        (eye_center_x, eye_center_y - cross_size), 
                        (eye_center_x, eye_center_y + cross_size), 
                        (0, 255, 0), 2)
            
            # Deteksi kedip berdasarkan jumlah mata
            if len(eyes) == 0:  # Mata tertutup
                self.eye_closed_frames += 1
                self.eye_open_frames = 0
                if self.eye_closed_frames >= 3 and not self.blink_detected:
                    self.blink_detected = True
                    return "KEDIP"
            elif len(eyes) >= 2:  # Mata terbuka
                self.eye_open_frames += 1
                if self.eye_open_frames >= 2:
                    self.eye_closed_frames = 0
                    self.blink_detected = False
            
            # Deteksi arah kepala sederhana berdasarkan posisi wajah
            frame_center_x = frame.shape[1] // 2
            face_center_x = x + w // 2
            offset = face_center_x - frame_center_x
            
            threshold = 80
            if offset < -threshold:
                return "LIHAT KANAN"
            elif offset > threshold:
                return "LIHAT KIRI"
            else:
                return "LIHAT TENGAH"
                
        except Exception as e:
            print(f"⚠️ OpenCV fallback error: {e}")
            return ""

    def detect_action(self, frame):
        """Deteksi aksi utama - frame akan dimodifikasi dengan tanda + pada pupil"""
        if self.use_gaze_tracking:
            return self.detect_action_gaze_tracking(frame)
        else:
            return self.detect_action_opencv_fallback(frame)

    def recognize_face(self, frame):
        """Mengenali wajah dan menampilkan label dari dataset"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            best_name = None
            best_confidence = 0
            best_location = None
            
            for top, right, bottom, left in face_locations:
                face_image = frame[top:bottom, left:right]
                
                if face_image.size == 0:
                    continue
                    
                # Preprocessing gambar wajah
                resized_face = cv2.resize(face_image, (224, 224))
                normalized_face = resized_face.astype("float32") / 255.0
                face_input = np.expand_dims(normalized_face, axis=0)
                
                # Prediksi dengan model
                interpreter = self.model
                interpreter.set_tensor(self.input_details[0]['index'], face_input.astype(np.float32))
                interpreter.invoke()
                predictions = interpreter.get_tensor(self.output_details[0]['index'])
                confidence = np.max(predictions)
                label_idx = np.argmax(predictions)
                
                # Dapatkan nama label dari dataset
                label_name = self.label_names.get(label_idx, f"Unknown-{label_idx}")
                
                if confidence > 0.7:  # Threshold lebih tinggi untuk keamanan
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_name = label_name
                        best_location = (left, top, right, bottom)
            
            return best_name, best_confidence, best_location
            
        except Exception as e:
            print(f"⚠️ Face recognition error: {e}")
            return None, 0, None

    def draw_verification_ui(self, frame):
        """Menggambar UI verifikasi"""
        try:
            # Background untuk text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Header
            cv2.putText(frame, "DOOR LOCK SECURITY SYSTEM - CONTINUOUS MODE", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 2)
            
            # Method info
            method_text = "Gaze Tracking" if self.use_gaze_tracking else "OpenCV Fallback"
            cv2.putText(frame, f"Method: {method_text}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Status tahap
            stage_text = ""
            if self.current_stage == "DETECTING_FACE":
                stage_text = "TAHAP 1: Mendeteksi wajah..."
            elif self.current_stage == "RECOGNIZING_FACE":
                stage_text = "TAHAP 2: Mengenali identitas..."
            elif self.current_stage == "GIVING_COMMAND":
                stage_text = f"TAHAP 3: Ikuti perintah -> {self.current_command}"
            elif self.current_stage == "VERIFYING_ACTION":
                stage_text = f"TAHAP 4: Memverifikasi -> {self.current_command}"
            elif self.current_stage == "SUCCESS":
                stage_text = "TAHAP 5: VERIFIKASI BERHASIL!"
            elif self.current_stage == "WAITING_RESET":
                stage_text = "MENUNGGU PENGGUNA BARU..."
            
            cv2.putText(frame, stage_text, (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            
            # Info wajah yang terdeteksi
            if self.detected_face_name:
                cv2.putText(frame, f"Wajah: {self.detected_face_name}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
            
            # Progress
            if self.current_stage in ["DETECTING_FACE", "RECOGNIZING_FACE"]:
                progress = min(self.face_stable_count / 10, 1.0) * 100
                cv2.putText(frame, f"Progress: {int(progress)}%", (20, 135), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif self.current_stage == "VERIFYING_ACTION":
                progress = min(self.action_stable_count / 5, 1.0) * 100
                cv2.putText(frame, f"Verifikasi: {int(progress)}%", (20, 135), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Waktu tersisa untuk sesi ini
            if self.current_stage not in ["SUCCESS", "WAITING_RESET"]:
                elapsed = time.time() - self.start_time
                remaining = max(0, self.timeout_seconds - elapsed)
                cv2.putText(frame, f"Waktu: {int(remaining)}s", (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Kontrol
            cv2.putText(frame, "Tekan 'Q' untuk keluar sistem", (20, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            print(f"⚠️ UI drawing error: {e}")

    def activate_relay(self):
        """Aktifkan relay untuk membuka solenoid pintu"""
        try:
            if self.gpio_controller:
                print("🔓 Mengaktifkan solenoid...")
                
                # Aktifkan relay dengan sysfs
                self.gpio_controller.on()  # GPIO HIGH = Relay ON
                print("✅ Relay ON - Solenoid aktif (pintu terbuka)")
                
                # Tahan solenoid aktif selama 10 detik
                time.sleep(10)
                
                # Matikan relay
                self.gpio_controller.off()  # GPIO LOW = Relay OFF
                print("🔒 Relay OFF - Solenoid nonaktif (pintu tertutup)")
                
            else:
                print("⚠️ Relay tidak tersedia, simulasi buka pintu")
                print("🔓 [SIMULASI] Solenoid aktif selama 10 detik")
                time.sleep(10)
                print("🔒 [SIMULASI] Solenoid nonaktif")
                
        except Exception as e:
            print(f"⚠️ Error mengaktifkan relay: {e}")
            # Pastikan relay dalam keadaan OFF jika ada error
            try:
                if self.gpio_controller:
                    self.gpio_controller.off()
            except:
                pass

    def process_verification_stages(self, frame):
        """Memproses tahap verifikasi"""
        try:
            # Jika dalam tahap WAITING_RESET, tunggu sampai tidak ada wajah terdeteksi
            if self.current_stage == "WAITING_RESET":
                # Cek apakah masih ada wajah
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 0:
                    # Tidak ada wajah, reset untuk sesi baru
                    print("🔄 Reset sistem untuk pengguna baru")
                    self.reset_verification_state()
                return
            
            # Deteksi wajah
            if int(time.time() * 10) % 3 == 0:  # Setiap 300ms
                self.last_face_result = self.recognize_face(frame)
            detected_name, confidence, face_location = self.last_face_result
                    
            # Deteksi aksi (ini akan menambahkan tanda + pada frame)
            detected_action = self.detect_action(frame)
            
            # Gambar bounding box jika ada wajah
            if face_location:
                left, top, right, bottom = face_location
                color = (0, 255, 0) if detected_name else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                if detected_name:
                    cv2.putText(frame, f"{detected_name} ({confidence:.2f})", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # TAHAP 1: DETECTING_FACE
            if self.current_stage == "DETECTING_FACE":
                if detected_name and confidence > 0.7:
                    self.face_stable_count += 1
                    self.detected_face_name = detected_name
                    
                    if self.face_stable_count >= 10:  # 10 frame stabil
                        print(f"✅ Wajah terdeteksi: {detected_name}")
                        self.current_stage = "RECOGNIZING_FACE"
                        self.face_stable_count = 0
                else:
                    self.face_stable_count = 0
                    self.detected_face_name = None
            
            # TAHAP 2: RECOGNIZING_FACE
            elif self.current_stage == "RECOGNIZING_FACE":
                if detected_name == self.detected_face_name and confidence > 0.7:
                    self.face_stable_count += 1
                    
                    if self.face_stable_count >= 5:  # 5 frame konfirmasi
                        print(f"✅ Identitas dikonfirmasi: {self.detected_face_name}")
                        self.current_stage = "GIVING_COMMAND"
                        self.current_command = random.choice(self.commands)
                        self.command_start_time = time.time()
                        print(f"🎯 Perintah: {self.current_command}")
                        self.face_stable_count = 0
                else:
                    self.current_stage = "DETECTING_FACE"
                    self.face_stable_count = 0
            
            # TAHAP 3: GIVING_COMMAND
            elif self.current_stage == "GIVING_COMMAND":
                # Timeout perintah - 8 detik
                if time.time() - self.command_start_time > 8:
                    self.current_command = random.choice(self.commands)
                    self.command_start_time = time.time()
                    print(f"🔄 Perintah baru: {self.current_command}")
                
                # Cek aksi user
                if detected_action == self.current_command:
                    print(f"✅ Aksi terdeteksi: {detected_action}")
                    self.current_stage = "VERIFYING_ACTION"
                    self.action_stable_count = 0
            
            # TAHAP 4: VERIFYING_ACTION
            elif self.current_stage == "VERIFYING_ACTION":
                if detected_action == self.current_command:
                    self.action_stable_count += 1
                    
                    if self.action_stable_count >= 5:  # 5 frame konsisten
                        print(f"🎉 VERIFIKASI BERHASIL untuk {self.detected_face_name}!")
                        self.activate_relay()  # Buka pintu
                        self.current_stage = "SUCCESS"
                        self.verification_complete = True
                        self.verification_success = True
                else:
                    self.current_stage = "GIVING_COMMAND"
                    self.action_stable_count = 0
                    self.command_start_time = time.time()
            
            # TAHAP 5: SUCCESS - Setelah 3 detik, pindah ke WAITING_RESET
            elif self.current_stage == "SUCCESS":
                if not hasattr(self, 'success_start_time'):
                    self.success_start_time = time.time()
                
                if time.time() - self.success_start_time > 3:
                    print("⏳ Menunggu pengguna pergi untuk reset sistem...")
                    self.current_stage = "WAITING_RESET"
                    delattr(self, 'success_start_time')
            
            # Tampilkan aksi yang terdeteksi
            if detected_action:
                cv2.putText(frame, f"Aksi: {detected_action}", (frame.shape[1] - 200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Cek timeout untuk sesi ini
            if self.current_stage not in ["SUCCESS", "WAITING_RESET"]:
                if time.time() - self.start_time > self.timeout_seconds:
                    print("⏰ Timeout sesi verifikasi, reset untuk pengguna baru")
                    self.reset_verification_state()
                           
        except Exception as e:
            print(f"⚠️ Error in verification stages: {e}")

    def run_continuous_system(self):
        """Menjalankan sistem dalam mode kontinyu"""
        print("🚀 MEMULAI SISTEM DOOR LOCK - MODE KONTINYU")
        print(f"Method: {'GazeTracking' if self.use_gaze_tracking else 'OpenCV Fallback'}")
        print("Sistem akan berjalan terus menerus dan otomatis reset setelah setiap verifikasi")
        print("Tahapan: Deteksi → Pengenalan → Perintah → Verifikasi → Sukses → Reset → Loop")
        print("Tekan 'Q' untuk menghentikan sistem\n")
        
        window_name = 'Door Lock Security System - Continuous'
        
        try:
            while self.system_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Gagal membaca frame kamera")
                    break
                
                # Proses verifikasi
                self.process_verification_stages(frame)
                
                # UI
                self.draw_verification_ui(frame)
                
                # Tampilkan dengan error handling
                try:
                    cv2.imshow(window_name, frame)
                except cv2.error as e:
                    print(f"OpenCV display error: {e}")
                    # Coba buat window baru
                    cv2.destroyAllWindows()
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    try:
                        cv2.imshow(window_name, frame)
                    except:
                        print("❌ Tidak dapat menampilkan window")
                        break
                
                # Input dengan timeout
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("❌ Sistem dihentikan oleh user")
                    self.system_running = False
                    break
                
                # Cek jika window ditutup
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("❌ Window ditutup")
                        break
                except cv2.error:
                    # Window sudah tidak ada
                    break
        
        except KeyboardInterrupt:
            print("❌ Interrupsi keyboard")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Membersihkan resources...")
        
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                print("✅ Kamera dilepas")
        except:
            pass
        
        try:
            # Cleanup GPIO menggunakan sysfs
            if hasattr(self, 'gpio_controller') and self.gpio_controller:
                self.gpio_controller.cleanup()
        except Exception as e:
            print(f"⚠️ Warning cleanup GPIO: {e}")

        # Cleanup OpenCV windows dengan safety
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            print("✅ OpenCV windows ditutup")
        except:
            pass
        
        print("✅ Cleanup selesai")

    def run_single_verification(self):
        """Menjalankan proses verifikasi single (seperti kode asli)"""
        print("🚀 MEMULAI SISTEM DOOR LOCK - MODE SINGLE")
        print(f"Method: {'GazeTracking' if self.use_gaze_tracking else 'OpenCV Fallback'}")
        print("Tahapan: Deteksi → Pengenalan → Perintah → Verifikasi → Sukses → Pintu Terbuka")
        print("Tekan 'Q' untuk membatalkan\n")
        
        window_name = 'Door Lock Security System'
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Gagal membaca frame kamera")
                    break
                
                # Cek timeout
                if time.time() - self.start_time > self.timeout_seconds:
                    print("⏰ Timeout verifikasi!")
                    break
                
                # Proses verifikasi - untuk single mode, tidak ada WAITING_RESET
                if self.current_stage != "WAITING_RESET":
                    self.process_verification_stages_single(frame)
                
                # UI
                self.draw_verification_ui_single(frame)
                
                # Tampilkan dengan error handling
                try:
                    cv2.imshow(window_name, frame)
                except cv2.error as e:
                    print(f"OpenCV display error: {e}")
                    # Coba buat window baru
                    cv2.destroyAllWindows()
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    try:
                        cv2.imshow(window_name, frame)
                    except:
                        print("❌ Tidak dapat menampilkan window")
                        break
                
                # Jika sukses, tahan 5 detik
                if self.current_stage == "SUCCESS":
                    time.sleep(5)
                    break
                
                # Input dengan timeout
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("❌ Verifikasi dibatalkan")
                    break
                
                # Cek jika window ditutup
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("❌ Window ditutup")
                        break
                except cv2.error:
                    # Window sudah tidak ada
                    break
        
        except KeyboardInterrupt:
            print("❌ Interupsi keyboard")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        finally:
            self.cleanup()
        
        return self.verification_success

    def process_verification_stages_single(self, frame):
        """Memproses tahap verifikasi untuk mode single (tanpa WAITING_RESET)"""
        try:
            # Deteksi wajah
            if int(time.time() * 10) % 3 == 0:  # Setiap 300ms
                self.last_face_result = self.recognize_face(frame)
            detected_name, confidence, face_location = self.last_face_result
                    
            # Deteksi aksi (ini akan menambahkan tanda + pada frame)
            detected_action = self.detect_action(frame)
            
            # Gambar bounding box jika ada wajah
            if face_location:
                left, top, right, bottom = face_location
                color = (0, 255, 0) if detected_name else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                if detected_name:
                    cv2.putText(frame, f"{detected_name} ({confidence:.2f})", (left, top - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # TAHAP 1: DETECTING_FACE
            if self.current_stage == "DETECTING_FACE":
                if detected_name and confidence > 0.7:
                    self.face_stable_count += 1
                    self.detected_face_name = detected_name
                    
                    if self.face_stable_count >= 10:  # 10 frame stabil
                        print(f"✅ Wajah terdeteksi: {detected_name}")
                        self.current_stage = "RECOGNIZING_FACE"
                        self.face_stable_count = 0
                else:
                    self.face_stable_count = 0
                    self.detected_face_name = None
            
            # TAHAP 2: RECOGNIZING_FACE
            elif self.current_stage == "RECOGNIZING_FACE":
                if detected_name == self.detected_face_name and confidence > 0.7:
                    self.face_stable_count += 1
                    
                    if self.face_stable_count >= 5:  # 5 frame konfirmasi
                        print(f"✅ Identitas dikonfirmasi: {self.detected_face_name}")
                        self.current_stage = "GIVING_COMMAND"
                        self.current_command = random.choice(self.commands)
                        self.command_start_time = time.time()
                        print(f"🎯 Perintah: {self.current_command}")
                        self.face_stable_count = 0
                else:
                    self.current_stage = "DETECTING_FACE"
                    self.face_stable_count = 0
            
            # TAHAP 3: GIVING_COMMAND
            elif self.current_stage == "GIVING_COMMAND":
                # Timeout perintah - 8 detik
                if time.time() - self.command_start_time > 8:
                    self.current_command = random.choice(self.commands)
                    self.command_start_time = time.time()
                    print(f"🔄 Perintah baru: {self.current_command}")
                
                # Cek aksi user
                if detected_action == self.current_command:
                    print(f"✅ Aksi terdeteksi: {detected_action}")
                    self.current_stage = "VERIFYING_ACTION"
                    self.action_stable_count = 0
            
            # TAHAP 4: VERIFYING_ACTION
            elif self.current_stage == "VERIFYING_ACTION":
                if detected_action == self.current_command:
                    self.action_stable_count += 1
                    
                    if self.action_stable_count >= 5:  # 5 frame konsisten
                        print(f"🎉 VERIFIKASI BERHASIL untuk {self.detected_face_name}!")
                        self.activate_relay()  # Buka pintu
                        self.current_stage = "SUCCESS"
                        self.verification_complete = True
                        self.verification_success = True
                else:
                    self.current_stage = "GIVING_COMMAND"
                    self.action_stable_count = 0
                    self.command_start_time = time.time()
            
            # Tampilkan aksi yang terdeteksi
            if detected_action:
                cv2.putText(frame, f"Aksi: {detected_action}", (frame.shape[1] - 200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                           
        except Exception as e:
            print(f"⚠️ Error in verification stages: {e}")

    def draw_verification_ui_single(self, frame):
        """Menggambar UI verifikasi untuk mode single"""
        try:
            # Background untuk text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Header
            cv2.putText(frame, "DOOR LOCK SECURITY SYSTEM", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            
            # Method info
            method_text = "Gaze Tracking" if self.use_gaze_tracking else "OpenCV Fallback"
            cv2.putText(frame, f"Method: {method_text}", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Status tahap
            stage_text = ""
            if self.current_stage == "DETECTING_FACE":
                stage_text = "TAHAP 1: Mendeteksi wajah..."
            elif self.current_stage == "RECOGNIZING_FACE":
                stage_text = "TAHAP 2: Mengenali identitas..."
            elif self.current_stage == "GIVING_COMMAND":
                stage_text = f"TAHAP 3: Ikuti perintah -> {self.current_command}"
            elif self.current_stage == "VERIFYING_ACTION":
                stage_text = f"TAHAP 4: Memverifikasi -> {self.current_command}"
            elif self.current_stage == "SUCCESS":
                stage_text = "TAHAP 5: VERIFIKASI BERHASIL!"
            
            cv2.putText(frame, stage_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            
            # Info wajah yang terdeteksi
            if self.detected_face_name:
                cv2.putText(frame, f"Wajah: {self.detected_face_name}", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
            
            # Progress
            if self.current_stage in ["DETECTING_FACE", "RECOGNIZING_FACE"]:
                progress = min(self.face_stable_count / 10, 1.0) * 100
                cv2.putText(frame, f"Progress: {int(progress)}%", (20, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif self.current_stage == "VERIFYING_ACTION":
                progress = min(self.action_stable_count / 5, 1.0) * 100
                cv2.putText(frame, f"Verifikasi: {int(progress)}%", (20, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Waktu tersisa
            elapsed = time.time() - self.start_time
            remaining = max(0, self.timeout_seconds - elapsed)
            cv2.putText(frame, f"Waktu: {int(remaining)}s", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Kontrol
            cv2.putText(frame, "Tekan 'Q' untuk batal", (20, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        except Exception as e:
            print(f"⚠️ UI drawing error: {e}")

# Fungsi untuk mode kontinyu
def run_continuous_door_lock():
    """
    Fungsi untuk menjalankan sistem dalam mode kontinyu
    Sistem akan terus berjalan dan reset otomatis setelah setiap verifikasi
    """
    try:
        print("Memulai sistem dalam mode kontinyu...")
        system = IntegratedVerificationSystem()
        system.run_continuous_system()
        
        print("🏁 Sistem door lock berhenti")
        return True
        
    except Exception as e:
        print(f"💥 Error sistem verifikasi kontinyu: {str(e)}")
        return False

# Fungsi utama untuk integrasi dengan GUI (mode single)
def run_complete_verification():
    """
    Fungsi untuk dipanggil dari main_window.py (mode single verification)
    Returns: True jika berhasil, False jika gagal
    """
    try:
        print("Memulai sistem mode single...")
        system = IntegratedVerificationSystem()
        
        # Mode single - hanya 1 kali verifikasi
        result = system.run_single_verification()
        
        if result:
            print("🎉 VERIFIKASI BERHASIL - Membuka pintu")
        else:
            print("❌ VERIFIKASI GAGAL")
            
        return result
        
    except Exception as e:
        print(f"💥 Error sistem verifikasi: {str(e)}")
        return False

    def run_single_verification(self):
        """Menjalankan proses verifikasi single (seperti kode asli)"""
        print("🚀 MEMULAI SISTEM DOOR LOCK - MODE SINGLE")
        print(f"Method: {'GazeTracking' if self.use_gaze_tracking else 'OpenCV Fallback'}")
        print("Tahapan: Deteksi → Pengenalan → Perintah → Verifikasi → Sukses → Pintu Terbuka")
        print("Tekan 'Q' untuk membatalkan\n")
        
        window_name = 'Door Lock Security System'
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Gagal membaca frame kamera")
                    break
                
                # Cek timeout
                if time.time() - self.start_time > self.timeout_seconds:
                    print("⏰ Timeout verifikasi!")
                    break
                
                # Proses verifikasi
                self.process_verification_stages(frame)
                
                # UI
                self.draw_verification_ui(frame)
                
                # Tampilkan dengan error handling
                try:
                    cv2.imshow(window_name, frame)
                except cv2.error as e:
                    print(f"OpenCV display error: {e}")
                    # Coba buat window baru
                    cv2.destroyAllWindows()
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    try:
                        cv2.imshow(window_name, frame)
                    except:
                        print("❌ Tidak dapat menampilkan window")
                        break
                
                # Jika sukses, tahan 3 detik
                if self.current_stage == "SUCCESS":
                    time.sleep(3)
                    break
                
                # Input dengan timeout
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("❌ Verifikasi dibatalkan")
                    break
                
                # Cek jika window ditutup
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("❌ Window ditutup")
                        break
                except cv2.error:
                    # Window sudah tidak ada
                    break
        
        except KeyboardInterrupt:
            print("❌ Interupsi keyboard")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        finally:
            self.cleanup()
        
        return self.verification_success

if __name__ == "__main__":
    # Menu pilihan
    print("=== DOOR LOCK SECURITY SYSTEM ===")
    print("1. Mode Single (1 kali verifikasi)")
    print("2. Mode Continuous (loop terus menerus)")
    
    choice = input("Pilih mode (1/2): ").strip()
    
    if choice == "1":
        print("\n--- MODE SINGLE ---")
        success = run_complete_verification()
        print(f"\nHasil: {'BERHASIL' if success else 'GAGAL'}")
    elif choice == "2":
        print("\n--- MODE CONTINUOUS ---")
        run_continuous_door_lock()
    else:
        print("Pilihan tidak valid!")
        exit(1)