# GPIO Control menggunakan lgpio untuk Raspberry Pi 5
import lgpio
import time

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

# Test fungsi
if __name__ == "__main__":
    try:
        print("Testing GPIO dengan lgpio...")
        gpio = GPIOController(18)
        
        print("Menghidupkan relay...")
        gpio.on()
        time.sleep(2)
        
        print("Mematikan relay...")
        gpio.off()
        time.sleep(1)
        
        gpio.cleanup()
        print("Test selesai!")
        
    except Exception as e:
        print(f"Error: {e}")