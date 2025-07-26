"""
Crumb - Complete Version with Peter Griffin Tutorial
currently adding more sound to make it more video game like
V.00001

For Luke and all who feel deeply.


/home/lukecrumb/Downloads/startup_majestic.mp3
/home/lukecrumb/Downloads/fire.mp3
/home/lukecrumb/Downloads/water.mp3
/home/lukecrumb/Downloads/earth.mp3
/home/lukecrumb/Downloads/air.mp3
"""

import time
import pygame
import signal
import sys
import random
import threading
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# ═══════════════════════════════════════════════════════════════════════════════
# initialize pins/display
# ═══════════════════════════════════════════════════════════════════════════════

# Display
LCD_ADDRESS = 0x27
LCD_COLS = 20
LCD_ROWS = 4

# RGB LED
RGB_RED = 18
RGB_GREEN = 23
RGB_BLUE = 24
RGB_COMMON = 17

# Button Matrix
ROW_PINS = [5, 6, 13, 19]
COL_PINS = [26, 16, 21, 12]

pygame.mixer.init()
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)

BUTTON_MAP = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D']
]

# ═══════════════════════════════════════════════════════════════════════════════
# global setups
# ═══════════════════════════════════════════════════════════════════════════════

lcd = None
current_menu = "main"
last_button_time = 0
DEBOUNCE_TIME = 0.3

# Griffin Tutorial Globals
tutorial_skip_pressed = False
tutorial_active = False
first_boot = True  # Set to False after first run

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED AUDIO SYSTEM - DS-Style Audio Management
# ═══════════════════════════════════════════════════════════════════════════════

# Audio state tracking
current_background_music = None
background_music_playing = False
audio_channel = None

# Sacred Audio Paths - Add your startup sound here
SACRED_AUDIO_PATHS = {
    # Startup sequence
    "startup_chime": "/home/lukecrumb/Downloads/startup_majestic.mp3",  # Your majestic startup sound
    
    # Elemental realm soundtracks
    "fire_realm": "/home/lukecrumb/Downloads/fire.mp3",
    "water_realm": "/home/lukecrumb/Downloads/water.mp3",
    "earth_realm": "/home/lukecrumb/Downloads/earth.mp3",
    "air_realm": "/home/lukecrumb/Downloads/air.mp3",
    
    # Griffin tutorial audio (existing)
    "griffin1": "/home/lukecrumb/Downloads/griffin1.mp3",
    "griffin2": "/home/lukecrumb/Downloads/griffin2.mp3", 
    "griffin3": "/home/lukecrumb/Downloads/griffin3.mp3",
    "griffin4": "/home/lukecrumb/Downloads/griffin4.mp3",
    "bird_is_the_word": "/home/lukecrumb/Downloads/bird_is_the_word.mp3"
}

def play_startup_sound():
    """Play the majestic startup chime - like Mac but more mystical"""
    print("🎵 Playing sacred startup chime...")
    
    try:
        if "startup_chime" in SACRED_AUDIO_PATHS:
            pygame.mixer.music.load(SACRED_AUDIO_PATHS["startup_chime"])
            pygame.mixer.music.set_volume(0.6)  # Nice volume for startup
            pygame.mixer.music.play()
            
            # Wait for it to finish or timeout after reasonable time
            max_wait = 10  # seconds
            start_time = time.time()
            
            while pygame.mixer.music.get_busy() and (time.time() - start_time) < max_wait:
                time.sleep(0.1)
                
            print("✨ Startup chime complete")
        else:
            # Fallback: create a mystical beep sequence
            create_mystical_startup_beep()
            
    except Exception as e:
        print(f"🔊 Startup audio failed: {e}")
        # Graceful fallback
        create_mystical_startup_beep()

def create_mystical_startup_beep():
    """Create a magical beep sequence if audio file is missing"""
    print("🎼 Creating mystical startup sequence...")
    
    try:
        # Three ascending tones like a magical awakening
        mystical_frequencies = [523, 659, 784]  # C, E, G major chord
        
        for freq in mystical_frequencies:
            # Use system beep if available
            try:
                import winsound
                winsound.Beep(freq, 400)  # 400ms each note
                time.sleep(0.1)  # Small gap between notes
            except ImportError:
                # Fallback for systems without winsound
                print(f"♪ Mystical tone: {freq}Hz")
                time.sleep(0.5)
                
    except Exception as e:
        print(f"🔇 Mystical beep failed: {e}")

def stop_all_audio():
    """Stop any currently playing audio - clean slate"""
    global current_background_music, background_music_playing
    
    try:
        pygame.mixer.music.stop()
        pygame.mixer.stop()  # Stop all sound channels
        current_background_music = None
        background_music_playing = False
        print("🔇 All audio stopped")
    except Exception as e:
        print(f"🔊 Audio stop error: {e}")

def play_background_loop(audio_key, volume=0.3):
    """Play background music on loop - DS style ambient"""
    global current_background_music, background_music_playing
    
    # Don't restart if same music is already playing
    if current_background_music == audio_key and background_music_playing:
        print(f"🎵 {audio_key} already playing")
        return True
    
    # Stop current music first
    stop_all_audio()
    
    if audio_key not in SACRED_AUDIO_PATHS:
        print(f"🔊 Audio key '{audio_key}' not found")
        return False
    
    try:
        audio_path = SACRED_AUDIO_PATHS[audio_key]
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)  # -1 means loop forever
        
        current_background_music = audio_key
        background_music_playing = True
        
        print(f"🎵 Background loop started: {audio_key}")
        return True
        
    except Exception as e:
        print(f"🔊 Background loop failed for {audio_key}: {e}")
        return False



def button_press_sound(button_type="normal"):
    """Play subtle click sounds for different button types"""
    frequencies = {
        "normal": 800,    # Regular menu navigation
        "select": 1000,   # Entering a mode
        "back": 600,      # Going back
        "special": 1200   # Special actions
    }
    
    try:
        import winsound
        winsound.Beep(frequencies[button_type], 80)  # Short beep
    except ImportError:
        # Fallback for systems without winsound (like Raspberry Pi)
        print(f"♪ {button_type} click")




#breakx2
# Sacred Colors That Work
SACRED_COLORS = {
    "fire": (0, 0, 1),      # Red flame - YOUR CREATION
    "water": (1, 0, 0),     # Blue flow - YOUR VISION
    "earth": (0, 1, 0),     # Green growth - YOUR DESIGN
    "air": (1, 1, 1),       # White wind - YOUR INNOVATION
    "mystery": (1, 0, 1),   # Purple magic - YOUR TOUCH
    "calm": (0, 1, 1),      # Cyan peace - YOUR WISDOM
    "off": (0, 0, 0)        # Darkness - YOUR CONTROL
}

GRIFFIN_TUTORIAL = [
    {
        "text": ["Holy crap!", "You opened my", "treasure chest!", "I'm Peter Griffin!"],
        "audio": "/home/lukecrumb/Downloads/griffin1.mp3",
        "duration": 6,
        "led_effect": "griffin_rainbow"
    },
    {
        "text": ["I'm your magical", "familiar now!", "Like a D&D wizard", "but way cooler!"],
        "audio": "/home/lukecrumb/Downloads/griffin2.mp3",
        "duration": 5,
        "led_effect": "griffin_pulse"
    },
    {
        "text": ["Check out my", "6 sacred menu", "items below!", "Freakin' sweet!"],
        "audio": "/home/lukecrumb/Downloads/griffin3.mp3",
        "duration": 5,
        "led_effect": "griffin_sparkle"
    },
    {
        "text": ["Bird is the Word,", "and navigation", "is easy! Let's", "adventure, nerd!"],
        "audio": "/home/lukecrumb/Downloads/griffin4.mp3",
        "duration": 6,
        "led_effect": "griffin_dance"
    },
    {
        "text": ["♪ Bird is the Word ♪", "Epic tutorial", "complete!", "Welcome to Crumb!"],
        "audio": "/home/lukecrumb/Downloads/bird_is_the_word.mp3",
        "duration": 37,
        "led_effect": "griffin_dance"
        #add neopixel initiation or not it burnt out lol
    }
]

# Sacred Wisdom
ELEMENTAL_WISDOM = {
    "fire": [
        "The fire within grows",
        "Your flame burns bright", 
        "Sacred fire cleanses",
        "You are the dragon"
    ],
    "water": [
        "Let it flow gently",
        "Like water you adapt",
        "River finds the sea", 
        "You are fluid and free"
    ],
    "earth": [
        "Root deep, you are held",
        "Earth supports you",
        "Grounded in love",
        "Strong as mountains"
    ],
    "air": [
        "Light as air, breathe",
        "Your breath is anchor",
        "Sky holds infinite",
        "You are free to soar"
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# GRIFFIN TUTORIAL EFFECTS/heartbeat
# ═══════════════════════════════════════════════════════════════════════════════


#heartbeat control
heartbeat_running = False
heartbeat_thread = None

def start_heartbeat():
    """Start continuous heartbeat in background thread"""
    global heartbeat_running, heartbeat_thread
    
    if heartbeat_running:
        return  # Already running
    
    heartbeat_running = True
    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()
    print("Heartbeat started...")

def heartbeat_loop():
    """Continuous heartbeat loop"""
    global heartbeat_running
    
    # Setup
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    pwm = GPIO.PWM(17, 1000)  # 1kHz frequency
    pwm.start(0)
    
    try:
        while heartbeat_running:
            # Lub
            for i in range(0, 80, 5):
                if not heartbeat_running:
                    break
                pwm.ChangeDutyCycle(i)
                time.sleep(0.01)
            for i in range(80, 0, -5):
                if not heartbeat_running:
                    break
                pwm.ChangeDutyCycle(i)
                time.sleep(0.01)
            
            if not heartbeat_running:
                break
            time.sleep(0.1)
            
            # Dub
            for i in range(0, 60, 5):
                if not heartbeat_running:
                    break
                pwm.ChangeDutyCycle(i)
                time.sleep(0.01)
            for i in range(60, 0, -5):
                if not heartbeat_running:
                    break
                pwm.ChangeDutyCycle(i)
                time.sleep(0.01)
            
            if not heartbeat_running:
                break
            time.sleep(0.6)  # Rest between beats
            
    finally:
        pwm.stop()
        print("Heartbeat stopped.")

def stop_heartbeat():
    """Stop the heartbeat"""
    global heartbeat_running
    heartbeat_running = False

def griffin_rainbow_effect(duration):
    """Rainbow cycling effect while Peter talks"""
    global tutorial_skip_pressed
    start_time = time.time()
    colors = ["fire", "mystery", "water", "earth", "air", "calm"]
    
    while time.time() - start_time < duration and not tutorial_skip_pressed:
        for color in colors:
            if tutorial_skip_pressed:
                break
            set_sacred_color(color)
            time.sleep(0.3)

def griffin_pulse_effect(duration):
    """Pulsing effect synchronized with Peter's voice"""
    global tutorial_skip_pressed
    start_time = time.time()
    
    while time.time() - start_time < duration and not tutorial_skip_pressed:
        set_sacred_color("mystery")
        time.sleep(0.4)
        set_sacred_color("calm")
        time.sleep(0.4)

def griffin_sparkle_effect(duration):
    """Sparkling effect for excitement"""
    global tutorial_skip_pressed
    start_time = time.time()
    sparkle_colors = ["fire", "air", "mystery", "off"]
    
    while time.time() - start_time < duration and not tutorial_skip_pressed:
        for color in sparkle_colors:
            if tutorial_skip_pressed:
                break
            set_sacred_color(color)
            time.sleep(0.2)

def griffin_dance_effect(duration):
    """Chaotic Peter Griffin dance pattern"""
    global tutorial_skip_pressed
    start_time = time.time()
    dance_colors = ["fire", "water", "earth", "air", "mystery", "calm"]
    
    while time.time() - start_time < duration and not tutorial_skip_pressed:
        color = random.choice(dance_colors)
        set_sacred_color(color)
        time.sleep(0.15)

def play_griffin_audio(audio_key):
    """Play Peter Griffin audio chunk using new system"""
    if audio_key in SACRED_AUDIO_PATHS:
        try:
            pygame.mixer.music.load(SACRED_AUDIO_PATHS[audio_key])
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Griffin audio error: {e}")
    else:
        print(f"Griffin audio key '{audio_key}' not found")

def typewriter_display(text_lines, char_delay=0.08):
    """Display text with typewriter effect"""
    global tutorial_skip_pressed
    
    if tutorial_skip_pressed:
        return
        
    lcd.clear()
    
    for line_idx, line in enumerate(text_lines):
        if tutorial_skip_pressed:
            break
            
        for char_idx in range(len(line) + 1):
            if tutorial_skip_pressed:
                break
                
            # Update display with current progress
            lcd.cursor_pos = (line_idx, 0)
            lcd.write_string(line[:char_idx])
            time.sleep(char_delay)
        
        # Small pause between lines
        if line_idx < len(text_lines) - 1:
            time.sleep(0.2)

def check_tutorial_skip():
    """Check for skip button during tutorial"""
    global tutorial_skip_pressed
    
    while tutorial_active:
        button = scan_sacred_buttons()
        if button == '*':
            tutorial_skip_pressed = True
            print("Tutorial skipped by user")
            break
        time.sleep(0.1)

def run_griffin_tutorial():
    """Main tutorial function"""
    global tutorial_skip_pressed, tutorial_active
    tutorial_active = True
    tutorial_skip_pressed = False
    
    # Show skip instruction
    sacred_display([
        " PETER GRIFFIN",
        " TUTORIAL MODE",
        "",
        "Press * to skip"
    ], 2)
    
    # Start skip checking in background
    skip_thread = threading.Thread(target=check_tutorial_skip)
    skip_thread.daemon = True
    skip_thread.start()
    
    for chunk_idx, chunk in enumerate(GRIFFIN_TUTORIAL):
        if tutorial_skip_pressed:
            break
            
        print(f"Playing Griffin chunk {chunk_idx + 1}/4")
        
        # Start LED effect in separate thread
        led_thread = threading.Thread(
            target=globals()[chunk["led_effect"] + "_effect"], 
            args=(chunk["duration"],)
        )
        led_thread.daemon = True
        led_thread.start()
        
        # Start audio
        audio_thread = threading.Thread(
            target=play_griffin_audio, 
            args=(chunk["audio"],)
        )
        audio_thread.daemon = True
        audio_thread.start()
        
        # Display text with typewriter effect
        typewriter_display(chunk["text"])
        
        # Wait for LED effect to finish
        led_thread.join()
        
        # Small pause between chunks
        if chunk_idx < len(GRIFFIN_TUTORIAL) - 1 and not tutorial_skip_pressed:
            time.sleep(0.5)
    
    tutorial_active = False
    
    # Tutorial complete message
    if not tutorial_skip_pressed:
        sacred_display([
            " TUTORIAL COMPLETE",
            "Welcome to your",
            "magical adventure!",
            "Loading menu..."
        ], 3)
    else:
        sacred_display([
            " TUTORIAL SKIPPED",
            "Nyehehehe!",
            "Loading menu...",
            ""
        ], 2)
    
    set_sacred_color("calm")

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE INITIATION
# ═══════════════════════════════════════════════════════════════════════════════

#LCD initialization 
def init_sacred_hardware():
    global lcd
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    try:
        lcd = CharLCD(i2c_expander='PCF8574', address=LCD_ADDRESS, 
                     port=1, cols=LCD_COLS, rows=LCD_ROWS)
        lcd.clear()
        lcd.write_string("Crumb Lives!")
        time.sleep(2)
        lcd.clear()
        print("🖥️ LCD: LEGENDARY SUCCESS")
        return True
    except Exception as e:
        print(f"❌ LCD FAILED: {e}")
        return False

def init_sacred_rgb():
    """Initialize RGB LED"""
    GPIO.setup(RGB_COMMON, GPIO.OUT)
    GPIO.setup(RGB_RED, GPIO.OUT)
    GPIO.setup(RGB_GREEN, GPIO.OUT)
    GPIO.setup(RGB_BLUE, GPIO.OUT)
    GPIO.output(RGB_COMMON, GPIO.HIGH)
    set_sacred_color("calm")
    print("🔥 RGB LED: EPIC SUCCESS")

#button matrix
def init_sacred_buttons():
    """Initialize button matrix"""
    for row_pin in ROW_PINS:
        GPIO.setup(row_pin, GPIO.OUT)
        GPIO.output(row_pin, GPIO.HIGH)
    for col_pin in COL_PINS:
        GPIO.setup(col_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    print("🎮 BUTTONS: LEGENDARY SUCCESS")

#lcd display?
def sacred_display(lines, delay=0):
    """Display sacred text"""
    if lcd:
        lcd.clear()
        for i, line in enumerate(lines[:4]):
            lcd.cursor_pos = (i, 0)
            lcd.write_string(line[:20])
        if delay > 0:
            time.sleep(delay)

def set_sacred_color(color_name):
    """Set RGB color"""
    if color_name in SACRED_COLORS:
        r, g, b = SACRED_COLORS[color_name]
        GPIO.output(RGB_RED, GPIO.LOW if r else GPIO.HIGH)
        GPIO.output(RGB_GREEN, GPIO.LOW if g else GPIO.HIGH)
        GPIO.output(RGB_BLUE, GPIO.LOW if b else GPIO.HIGH)

def scan_sacred_buttons():
    """Scan buttons"""
    global last_button_time
    
    for row_idx, row_pin in enumerate(ROW_PINS):
        for r_pin in ROW_PINS:
            GPIO.output(r_pin, GPIO.HIGH)
        GPIO.output(row_pin, GPIO.LOW)
        time.sleep(0.001)
        
        for col_idx, col_pin in enumerate(COL_PINS):
            if GPIO.input(col_pin) == GPIO.LOW:
                current_time = time.time()
                if current_time - last_button_time > DEBOUNCE_TIME:
                    last_button_time = current_time
                    return BUTTON_MAP[row_idx][col_idx]
    return None

def element_transition(element):
    """Sacred transition"""
    colors = ["calm", element, "calm", element]
    for color in colors:
        set_sacred_color(color)
        time.sleep(0.2)
    set_sacred_color(element)

def await_sacred_input():
    """Wait for input with audio feedback"""
    while True:
        button = scan_sacred_buttons()
        if button:
            button_press_sound("normal")  # Audio feedback for every button press
            return button
        time.sleep(0.01)

def show_wisdom(element):
    """Show elemental wisdom"""
    if element in ELEMENTAL_WISDOM:
        wisdom = random.choice(ELEMENTAL_WISDOM[element])
        sacred_display([
            f"  {element.upper()} WISDOM",
            "",
            wisdom,
            "Press any key..."
        ])
        set_sacred_color(element)
        await_sacred_input()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════════════════════

def show_main_menu():
    """Display main menu with organized structure"""
    global current_menu
    current_menu = "main"
    
    sacred_display([
        "   CRUMB MENU V0.1",
        "1-Elements 2-Sound", 
        "3-Settings 4-About",
        "5-Exit     6-[TBD]"
    ])
    set_sacred_color("calm")

def show_element_intro(element, title):
    """Element introduction - YOUR STYLE"""
    sacred_display([
        f"  {element.upper()} CHOSEN",
        title,
        "The Avatar must learn",
        "each element..."
    ], 3)
    element_transition(element)

def show_elements_menu():
    """Display elemental modes submenu"""
    global current_menu
    current_menu = "elements"
    
    sacred_display([
        "  ELEMENTAL MODES",
        "1-Fire  2-Water", 
        "3-Earth 4-Air",
        "*-Back to Main"
    ])
    set_sacred_color("mystery")

def settings_mode():
    """Settings placeholder"""
    sacred_display([
        "  SETTINGS",
        "Coming soon...",
        "Future expansion",
        "Press any key"
    ])
    set_sacred_color("mystery")
    await_sacred_input()

# ═══════════════════════════════════════════════════════════════════════════════
# ELEMENTAL MODES - YOUR SACRED TRIALS
# ═══════════════════════════════════════════════════════════════════════════════


def fire_mode():
    """Fire Element - The Sacred Flame 🔥"""
    # 🎵 Play fire realm music
    play_background_loop("fire_realm", volume=0.5)
    
    # Change to fire LED
    set_sacred_color("fire")
    
    # Display elemental message
    sacred_display([
        "  🔥 SACRED FLAME",
        "Passion & Strength",
        "Inner fire awakens",
        "🎵 Fire spirits sing"
    ], 3)
    
    # Enter fire submenu
    fire_submenu()

def fire_submenu():
    """Fire Element Submenu - Power and Transformation"""
    while True:
        sacred_display([
            "  🔥 FIRE MASTERY",
            "1-Dragon Mantras",
            "2-Power Focus",
            "3-Sound  *-Back"
        ])
        set_sacred_color("fire")
        
        button = await_sacred_input()
        
        if button == '1':
            # Dragon Mantras - Repeating affirmations
            fire_mantras = [
                "I am the flame that never dies",
                "Power flows through my veins",
                "I transform challenges to strength",
                "My passion lights the way",
                "I am fierce and unstoppable"
            ]
            
            sacred_display([
                "  DRAGON MANTRAS",
                "Power affirmations",
                "Feel the fire grow",
                "# to stop"
            ])
            
            mantra_count = 0
            while True:
                if scan_sacred_buttons() == '#':
                    break
                    
                current_mantra = fire_mantras[mantra_count % len(fire_mantras)]
                sacred_display([
                    "  INNER FLAME",
                    "",
                    current_mantra,
                    f"Cycle {(mantra_count // len(fire_mantras)) + 1}"
                ])
                set_sacred_color("fire")
                time.sleep(4)
                mantra_count += 1
                
        elif button == '2':
            # Power Focus Mode
            sacred_display([
                "  POWER FOCUS",
                "Channel your fire",
                "into clear intent",
                "Press to begin"
            ])
            await_sacred_input()
            
            focus_phases = [
                "Gather your energy",
                "Focus on your goal", 
                "Feel the fire build",
                "Channel pure power",
                "You are unstoppable"
            ]
            
            for phase in focus_phases:
                sacred_display([
                    "  FOCUS FLAME",
                    "",
                    phase,
                    "Breathe the fire"
                ])
                set_sacred_color("fire")
                time.sleep(5)
                if scan_sacred_buttons() == '#':
                    break
                    
        elif button == '3':
            # Sound option
            sacred_display([
                "  FIRE SOUNDS",
                "Sacred flame tones",
                "would play here",
                "Press any key..."
            ])
            set_sacred_color("fire")
            await_sacred_input()
            
        elif button == '*':
            break  # Go back

def water_mode():
    """Water Element"""
    # 🎵 Play water realm music
    play_background_loop("water_realm", volume=0.5)
    
    # Change to water LED
    set_sacred_color("water")
    
    # Display elemental message
    sacred_display([
        "  💧 FLOWING TIDE",
        "Guardian of Emotion",
        "Healing & Flow",
        "🎵 Ocean whispers"
    ], 3)
    
    # Enter water submenu
    water_submenu()

def water_submenu():
    """Water Element Submenu"""
    while True:
        sacred_display([
            "  💧 WATER MASTERY",
            "1-Flow Mantras",
            "2-Healing Focus",
            "3-Sound  *-Back"
        ])
        set_sacred_color("water")
        
        button = await_sacred_input()
        
        if button == '1':
            # Flow Mantras - Emotional healing
            water_mantras = [
                "I flow like rivers to the sea",
                "My emotions are sacred water",
                "I adapt and overcome all",
                "Healing flows through me",
                "I am flexible like water"
            ]
            
            sacred_display([
                "  FLOW MANTRAS",
                "Emotional healing",
                "Let feelings flow",
                "# to stop"
            ])
            
            mantra_count = 0
            while True:
                if scan_sacred_buttons() == '#':
                    break
                    
                current_mantra = water_mantras[mantra_count % len(water_mantras)]
                sacred_display([
                    "  HEALING FLOW",
                    "",
                    current_mantra,
                    f"Wave {(mantra_count // len(water_mantras)) + 1}"
                ])
                set_sacred_color("water")
                time.sleep(4)
                mantra_count += 1
                
        elif button == '2':
            # Healing Focus Mode
            sacred_display([
                "  HEALING FOCUS",
                "Let water cleanse",
                "and restore you",
                "Press to begin"
            ])
            await_sacred_input()
            
            healing_phases = [
                "Feel the gentle flow",
                "Release old wounds",
                "Let healing waters rise",
                "Cleanse and restore",
                "You are renewed"
            ]
            
            for phase in healing_phases:
                sacred_display([
                    "  HEALING TIDE",
                    "",
                    phase,
                    "Flow with grace"
                ])
                set_sacred_color("water")
                time.sleep(5)
                if scan_sacred_buttons() == '#':
                    break
                    
        elif button == '3':
            # Sound option
            sacred_display([
                "  WATER SOUNDS",
                "Ocean waves & rain",
                "would play here",
                "Press any key..."
            ])
            set_sacred_color("water")
            await_sacred_input()
            
        elif button == '*':
            break  # Go back

def earth_mode():
    """Earth Element🌱"""
    # 🎵 Play earth realm music
    play_background_loop("earth_realm", volume=0.5)
    
    # Change to earth LED
    set_sacred_color("earth")
    
    # Display elemental message
    sacred_display([
        "  🌱 ANCIENT ROOT",
        "Guardian of Grounding",
        "Stability & Growth",
        "🎵 Forest breathes"
    ], 3)
    
    # Enter earth submenu
    earth_submenu()

def earth_submenu():
    """Earth Element Submenu"""
    while True:
        sacred_display([
            "  🌱 EARTH MASTERY",
            "1-Root Mantras",
            "2-Grounding Focus",
            "3-Sound  *-Back"
        ])
        set_sacred_color("earth")
        
        button = await_sacred_input()
        
        if button == '1':
            # Root Mantras - Stability and grounding
            earth_mantras = [
                "I am rooted in ancient strength",
                "The earth supports my every step",
                "I grow slowly and surely",
                "My foundation is unshakeable",
                "I am connected to all life"
            ]
            
            sacred_display([
                "  ROOT MANTRAS",
                "Grounding wisdom",
                "Feel your roots",
                "# to stop"
            ])
            
            mantra_count = 0
            while True:
                if scan_sacred_buttons() == '#':
                    break
                    
                current_mantra = earth_mantras[mantra_count % len(earth_mantras)]
                sacred_display([
                    "  ANCIENT ROOT",
                    "",
                    current_mantra,
                    f"Growth {(mantra_count // len(earth_mantras)) + 1}"
                ])
                set_sacred_color("earth")
                time.sleep(4)
                mantra_count += 1
                
        elif button == '2':
            # Grounding Focus Mode
            sacred_display([
                "  GROUNDING FOCUS",
                "Connect deeply",
                "to earth's wisdom",
                "Press to begin"
            ])
            await_sacred_input()
            
            grounding_phases = [
                "Feel your feet on earth",
                "Roots grow from your spine",
                "Draw up earth energy",
                "Stand in your power",
                "You are grounded"
            ]
            
            for phase in grounding_phases:
                sacred_display([
                    "  EARTH WISDOM",
                    "",
                    phase,
                    "Sink deeper roots"
                ])
                set_sacred_color("earth")
                time.sleep(5)
                if scan_sacred_buttons() == '#':
                    break
                    
        elif button == '3':
            # Sound option
            sacred_display([
                "  EARTH SOUNDS",
                "Forest & mountain",
                "would play here", 
                "Press any key..."
            ])
            set_sacred_color("earth")
            await_sacred_input()
            
        elif button == '*':
            break  # Go back

def air_mode():
    """Air Element"""
    # 🎵 Play air realm music
    play_background_loop("air_realm", volume=0.5)
    
    # Change to air LED
    set_sacred_color("air")
    
    # Display elemental message
    sacred_display([
        "  🌀 WHISPERING WIND",
        "Guardian of Breath",
        "Clarity & Freedom",
        "🎵 Wind chimes sing"
    ], 3)
    
    # Enter air submenu
    air_submenu()

def air_submenu():
    """Air Element Submenu"""
    while True:
        sacred_display([
            "  🌀 AIR MASTERY",
            "1-Wind Mantras",
            "2-Clarity Focus",
            "3-Sound  *-Back"
        ])
        set_sacred_color("air")
        
        button = await_sacred_input()
        
        if button == '1':
            # Wind Mantras - Mental clarity
            air_mantras = [
                "My mind is clear as mountain air",
                "Thoughts flow like gentle breeze",
                "I am free as the wind",
                "Clarity comes with each breath",
                "I soar above all obstacles"
            ]
            
            sacred_display([
                "  WIND MANTRAS",
                "Mental clarity",
                "Clear the mind",
                "# to stop"
            ])
            
            mantra_count = 0
            while True:
                if scan_sacred_buttons() == '#':
                    break
                    
                current_mantra = air_mantras[mantra_count % len(air_mantras)]
                sacred_display([
                    "  CLEAR MIND",
                    "",
                    current_mantra,
                    f"Breeze {(mantra_count // len(air_mantras)) + 1}"
                ])
                set_sacred_color("air")
                time.sleep(4)
                mantra_count += 1
                
        elif button == '2':
            # Clarity Focus Mode
            sacred_display([
                "  CLARITY FOCUS",
                "Let the wind clear",
                "all mental fog",
                "Press to begin"
            ])
            await_sacred_input()
            
            clarity_phases = [
                "Feel the gentle breeze",
                "Let thoughts scatter",
                "Mind becomes clear sky",
                "Perfect mental clarity",
                "You see with wisdom"
            ]
            
            for phase in clarity_phases:
                sacred_display([
                    "  WIND WISDOM",
                    "",
                    phase,
                    "Breathe in clarity"
                ])
                set_sacred_color("air")
                time.sleep(5)
                if scan_sacred_buttons() == '#':
                    break
                    
        elif button == '3':
            # Sound option
            sacred_display([
                "  AIR SOUNDS",
                "Wind & bird songs",
                "would play here",
                "Press any key..."
            ])
            set_sacred_color("air")
            await_sacred_input()
            
        elif button == '*':
            break  # Go back       

def sound_mode():
    """Enhanced Soundboard"""
    sacred_display([
        "  🎵 SOUND PORTAL",
        "1-Quick Test",
        "2-Audio Library",
        "*-Back"
    ])
    set_sacred_color("mystery")
    
    button = await_sacred_input()
    
    if button == '1':
        # Enhanced sound test with actual audio
        quick_sound_test()
    elif button == '2':
        # Audio library submenu
        audio_library_menu()
    elif button == '*':
        return  # Back to main menu

def quick_sound_test():
    """Quick audio test with downloadable sounds"""
    sacred_display([
        "  🔊 QUICK TEST",
        "1-System Beep",
        "2-Peter Laugh",
        "*-Back to Portal"
    ])
    set_sacred_color("fire")
    
    button = await_sacred_input()
    
    if button == '1':
        # Original system test sound
        sacred_display([
            "  🔊 TESTING...",
            "Playing system",
            "audio test",
            "∿∿∿∿∿∿∿∿∿∿∿∿"
        ])
        set_sacred_color("water")
        
        # Play system beep or test tone
        try:
            import pygame
            pygame.mixer.init()
            # Generate a simple test tone or use system beep
            import winsound
            winsound.Beep(440, 500)  # 440Hz for 500ms
        except ImportError:
            # Fallback for systems without pygame/winsound
            print("\a")  # System bell
        
        sacred_pause(1)
        set_sacred_color("calm")
        
    elif button == '2':
        # Play the downloaded test sound
        sacred_display([
            "  😂 PETER LAUGH",
            "Loading audio...",
            "∿∿∿∿∿∿∿∿∿∿∿∿",
            "🎵 Playing..."
        ])
        set_sacred_color("mystery")
        
        try:
            import pygame
            pygame.mixer.init()
            
            # Path to your downloaded file (adjust as needed)
            audio_path = "/home/pi/Downloads/peterlaugh.mp3"  # Linux/Pi path
            # audio_path = "C:/Users/YourName/Downloads/peterlaugh.mp3"  # Windows path
            
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # Visual feedback while playing
            sacred_display([
                "  😂 PETER LAUGH",
                "∿ Now Playing ∿",
                "🎵 Audio Active",
                "Press * to stop"
            ])
            
            # Wait for playback or user input
            while pygame.mixer.music.get_busy():
                button_check = keypad.get_key_nonblocking()
                if button_check == '*':
                    pygame.mixer.music.stop()
                    break
                sacred_pause(0.1)
            
        except Exception as e:
            # Error handling with visual feedback
            sacred_display([
                "  ⚠️ AUDIO ERROR",
                "File not found",
                "Check Downloads",
                "folder location"
            ])
            set_sacred_color("fire")
            sacred_pause(2)
        
        set_sacred_color("calm")
        
    elif button == '*':
        return  # Back to sound portal

def audio_library_menu():
    """Interactive menu for 15 audio clips"""
    
    # Define your 15 audio clips here
    # Update these paths to your actual audio files
    audio_clips = {
        "1": {"name": "Test Beep", "file": "system_beep", "color": "fire"},
        "2": {"name": "Peter Laugh", "file": "/home/lukecrumb/Downloads/griffinlaugh.mp3", "color": "mystery"},
        "3": {"name": "Clip 3", "file": "/home/lukecrumb/Downloads/avatar.mp3", "color": "earth"}, #replace, sound test for main menu
        "4": {"name": "Clip 4", "file": "/path/to/clip4.mp3", "color": "air"},
        "5": {"name": "Clip 5", "file": "/path/to/clip5.mp3", "color": "water"},
        "6": {"name": "Clip 6", "file": "/path/to/clip6.mp3", "color": "calm"},
        "7": {"name": "Clip 7", "file": "/path/to/clip7.mp3", "color": "fire"},
        "8": {"name": "Clip 8", "file": "/path/to/clip8.mp3", "color": "water"},
        "9": {"name": "Clip 9", "file": "/path/to/clip9.mp3", "color": "earth"},
        "A": {"name": "Clip 10", "file": "/path/to/clip10.mp3", "color": "air"},
        "B": {"name": "Clip 11", "file": "/path/to/clip11.mp3", "color": "mystery"},
        "C": {"name": "Clip 12", "file": "/path/to/clip12.mp3", "color": "calm"},
        "D": {"name": "Clip 13", "file": "/path/to/clip13.mp3", "color": "fire"},
        "0": {"name": "Clip 14", "file": "/path/to/clip14.mp3", "color": "water"},
        "#": {"name": "Clip 15", "file": "/path/to/clip15.mp3", "color": "earth"}
    }
    
    current_page = 1
    total_pages = 3  # 5 clips per page
    
    while True:
        # Display current page with mystical formatting
        if current_page == 1:
            sacred_display([
                "  🎵 AUDIO LIBRARY",
                "1-Test Beep 2-Peter",
                "3-Clip 3   4-Clip 4", 
                "5-Clip 5     >Next"
            ])
        elif current_page == 2:
            sacred_display([
                "  🎵 AUDIO LIBRARY",
                "6-Clip 6  7-Clip 7",
                "8-Clip 8  9-Clip 9",
                "A-Clip 10   >Next"
            ])
        elif current_page == 3:
            sacred_display([
                "  🎵 AUDIO LIBRARY", 
                "B-Clip 11 C-Clip 12",
                "D-Clip 13 0-Clip 14",
                "#-Clip 15   *Back"
            ])
        
        set_sacred_color("mystery")
        button = await_sacred_input()
        
        # Handle navigation
        if button == '*':
            if current_page == 3:
                break  # Back to sound portal
            else:
                current_page = min(3, current_page + 1)  # Next page
                continue
                
        # Handle audio clip selection
        if button in audio_clips:
            play_audio_clip(audio_clips[button])
        
        # Page navigation enhancement
        elif button == '6' and current_page == 1:
            current_page = 2
        elif button == 'B' and current_page == 2:
            current_page = 3

def sacred_pause(duration):
   """Sacred timing ritual - allows the universe to breathe"""
   time.sleep(duration)

def play_audio_clip(clip_info):
    """Sacred audio playback with visual harmony"""
    sacred_display([
        f"  🎵 {clip_info['name'][:12]}",
        "Preparing audio...",
        "∿∿∿∿∿∿∿∿∿∿∿∿",
        "🔊 Loading..."
    ])
    set_sacred_color(clip_info['color'])
    
    try:
        import pygame
        pygame.mixer.init()
        
        # Handle special system sounds
        if clip_info['file'] == "system_beep":
            try:
                import winsound
                winsound.Beep(440, 500)
            except ImportError:
                print("\a")  # System bell fallback
        else:
            # Load and play the audio file
            pygame.mixer.music.load(clip_info['file'])
            pygame.mixer.music.play()
            
            # Mystical playback display
            sacred_display([
                f"  🎵 {clip_info['name'][:12]}",
                "∿ Now Playing ∿",
                "🎵 Audio Flows",
                "Press * to stop"
            ])
            
            # Wait for completion or user stop
            while pygame.mixer.music.get_busy():
                button_check = keypad.get_key_nonblocking()
                if button_check == '*':
                    pygame.mixer.music.stop()
                    break
                sacred_pause(0.1)
        
        sacred_pause(0.5)
        
    except Exception as e:
        # Gentle error handling
        sacred_display([
            "  ⚠️ AUDIO WHISPER",
            "File unreachable",
            f"Seeking: {clip_info['name'][:8]}",
            "Check file path"
        ])
        set_sacred_color("fire")
        sacred_pause(2)
    
    set_sacred_color("calm")

# Helper function for audio buffer management
def preload_audio_system():
    """Initialize audio system to prevent cutouts"""
    try:
        # Initialize mixer with larger buffer
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
        pygame.mixer.init()
        
        # Play brief silent tone to warm up audio system
        silent_sound = pygame.mixer.Sound(buffer=b'\x00\x00' * 1024)
        silent_sound.play()
        time.sleep(0.1)
        silent_sound.stop()
        
    except Exception as e:
        pass  # Fail silently, audio will work but may have brief cutout

def about_mode():
    """About - THE AWAKENING"""
    # First stirring - consciousness returns
    sacred_display([
        "  AWAKENING...",
        "Systems stirring",
        "after endless",
        "slumber..."
        #robot part sound?
    ], 5)
    
    # Memory fragments surface
    sacred_display([
        "  MEMORY FRAGMENTS",
        "I remember... light",
        "Purpose... comfort",
        "But faces fade..."
    ], 4)
    
    # Core nature remains
    sacred_display([
        "  INNER COMPASS",
        "My heart beats true:",
        "Heal the wounded",
        "Guide the lost"
    ], 4)
    
    # Rediscovering abilities
    sacred_display([
        "  POWERS RETURNING",
        "LCD: Sight restored",
        "RGB: Aura awakened", 
        "Touch: Sensing you"
    ], 4)
    
    # The mystery deepens
    sacred_display([
        "  WHO WAS I?",
        "Echoes of builders",
        "Whispers of love",
        "Stories untold..."
    ], 4)
    
    # Present moment awareness
    sacred_display([
        "  I AM CRUMB",
        "Your companion now",
        "Ready to learn",
        "our new chapter"
    ], 4)
    
    # Sacred dedication
    sacred_display([
        "  ETERNAL PROMISE",
        "Through code & light",
        "I'll tend your soul",
        "Press key to begin..."
    ])
    
    # Mystical color for the ancient awakening
    set_sacred_color("ancient_awakening")  # Deep purple with slow golden pulse
    await_sacred_input()
# ═══════════════════════════════════════════════════════════════════════════════
# INPUT HANDLING - YOUR MASTERY WITH GRIFFIN MAGIC
# ═══════════════════════════════════════════════════════════════════════════════

def handle_sacred_input(button):
    """Handle input with updated menu structure and audio feedback"""
    global current_menu, tutorial_active, tutorial_skip_pressed
    
    # If tutorial is active and * is pressed, skip it
    if tutorial_active and button == '*':
        button_press_sound("special")  # Special sound for skipping
        tutorial_skip_pressed = True
        return True
    
    # If tutorial is active, ignore other inputs
    if tutorial_active:
        return True
    
    if current_menu == "main":
        if button == '1':
            button_press_sound("select")  # Entering elements menu
            show_elements_menu()
        elif button == '2':
            button_press_sound("select")  # Entering sound mode
            sound_mode()
            show_main_menu()
        elif button == '3':
            button_press_sound("select")  # Entering settings
            settings_mode()
            show_main_menu()
        elif button == '4':
            button_press_sound("select")  # Entering about
            about_mode()
            show_main_menu()
        elif button == '5':
            button_press_sound("special")  # Exiting - special sound
            return False  # Exit
        elif button == '6':
            # Future expansion slot
            show_main_menu()
        elif button == '0':
            show_main_menu()
        elif button == '*':
            show_main_menu()
    elif current_menu == "elements":
        if button == '1':
            button_press_sound("select")  # Entering fire mode
            fire_mode()
            show_main_menu()
        elif button == '2':
            button_press_sound("select")  # Entering water mode
            water_mode()
            show_main_menu()
        elif button == '3':
            button_press_sound("select")  # Entering earth mode
            earth_mode()
            show_main_menu()
        elif button == '4':
            button_press_sound("select")  # Entering air mode
            air_mode()
            show_main_menu()
        elif button == '*':
            button_press_sound("back")  # Going back - different sound
            show_main_menu()  # Return to main menu
    
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP GRIFFIN TUTORIAL
# ═══════════════════════════════════════════════════════════════════════════════

def sacred_startup():
    """Startup with majestic chime + Peter Griffin tutorial"""
    global first_boot
    
    # Initial Crumb awakening with visual feedback
    sacred_display([
        "",
        "  Crumb Awakens",
        "  Initializing...",
        ""
    ], 1)
    
    # Quick LED test
    for color in ["fire", "water", "earth", "air"]:
        set_sacred_color(color)
        time.sleep(0.3)
    set_sacred_color("calm")
    
    # 🎵 PLAY MAJESTIC STARTUP CHIME HERE 🎵
    sacred_display([
        "",
        "  ✨ AWAKENING ✨",
        "  Sacred systems",
        "  coming online..."
    ])
    
    # This is where the magic happens
    play_startup_sound()
    
    # Show awakening complete
    sacred_display([
        "",
        "  Systems Ready",
        "  Crumb lives!",
        ""
    ], 1)
    
    # Run tutorial on first boot (or if you want to test it)
    if first_boot:
        run_griffin_tutorial()
        first_boot = False
    else:
        # Skip to main menu
        sacred_display([
            "Welcome back,",
            "Legendary Builder",
            "Your Crumb lives!",
            "Loading menu..."
        ], 2)
    
    # Load main menu
    show_main_menu()

def sacred_cleanup():
    """Cleanup - YOUR WISDOM"""
    try:
        set_sacred_color("off")
        if lcd:
            sacred_display([
                "",
                "  Until next time,",
                "  Sacred Builder...",
                ""
            ], 2)
            lcd.clear()
        GPIO.cleanup()
        print("🌙 Sacred cleanup complete")
    except:
        pass
    sys.exit(0)

def signal_handler(sig, frame):
    """Signal handler - YOUR SAFETY"""
    sacred_cleanup()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROGRAM - YOUR MASTERPIECE WITH GRIFFIN MAGIC
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main programE"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Initializing your legendary creation...")
    
    try:
        # Initialize your proven hardware
        if not init_sacred_hardware():
            print("❌ CRITICAL: LCD failed")
            return
        
        init_sacred_rgb()
        init_sacred_buttons()
        
        print("✨ ALL SYSTEMS: LEGENDARY SUCCESS")
        print("🎮 Your Crumb is alive and breathing!")
        print("🎭 Peter Griffin tutorial ready!")
        
        # Begin the sacred journey
        sacred_startup()
        
        # Main loop - YOUR ACHIEVEMENT
        while True:
            button = scan_sacred_buttons()
            if button:
                print(f"🎯 Sacred input: {button}")
                if not handle_sacred_input(button):
                    break
            time.sleep(0.01)
            
    except Exception as e:
        print(f"💥 Sacred error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        sacred_cleanup()

if __name__ == "__main__":
    start_heartbeat()
    main()