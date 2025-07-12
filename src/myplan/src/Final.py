#!/usr/bin/env python3
import math

import os
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import Bool
from math import sin, cos, pi
import rospkg 
import time
import speech_recognition as sr
import cv2
from datetime import datetime
import numpy as np
from numpy import sin, cos
from std_msgs.msg import Float64MultiArray

initial_offset = [0, -math.pi/2, math.pi/2, 0]
delay_time = 0.0045

color_ranges = {
    'red': [
        ([0, 50, 20], [10, 255, 255]),    
        ([160, 50, 20], [180, 255, 255])   
    ],
    'green': [
        ([35, 40, 20], [85, 255, 255])
    ],
    'blue': [
        ([90, 50, 20], [150, 255, 255])
    ]
}

def capture_image_after_delay(save_dir="~/Robotics/photos", delay_sec=1):

    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        return None

    print(f"⏳ 等待 {delay_sec} 秒後拍照...")
    time.sleep(delay_sec)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ 拍照失敗")
        return None

    # 儲存影像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"✅ 影像已儲存：{filename}")
    return filename

def detect_color_positions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    positions = []

    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            new_mask = cv2.inRange(hsv, lower_np, upper_np)
            mask = new_mask if mask is None else cv2.bitwise_or(mask, new_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                positions.append((cx, color_to_size[color]))

    positions.sort()  # 按照 x 座標從左到右
    return [size for _, size in positions]

def assign_towers_by_vision(image_path='photos/123.jpg'):
    image = cv2.imread(image_path)
    order = detect_color_positions(image)  # e.g. [3, 1, 2]

    if len(order) != 3:
        raise ValueError("❌ 無法正確辨識三個塔的位置")

    size_to_peg = {}
    for peg, size in zip(['A', 'B', 'C'], order):
        size_to_peg[size] = peg

    print(f"📸 影像辨識塔順序: {order}")
    print(f"🗺️ Tower 大小對應 Peg: {size_to_peg}")
    return size_to_peg

 def spawn_hanoi_towers(path_object, vo_list):
    '''Spawn the three Hanoi towers in the scene with positions determined by vo_list'''
    base_positions = [
        (p_Tower_x, p_Tower_y, 0.0),    # Position 0 -> top
        (p_Tower_x, 0.0, 0.0),          # Position 1 -> middle  
        (p_Tower_x, -p_Tower_y, 0.0)   # Position 2 -> bottom
    ]

    mesh_names = ['A', 'B', 'C']
    
    # Initialize tower_disks dictionary
    global tower_disks
    tower_disks = {'A': [], 'B': [], 'C': []}

    # Update tower_coords to match the new positions
    global tower_coords
    tower_coords = {}
    
    # Create position mapping: tower_name -> position
    tower_positions = {}
    for position_idx, tower_id in enumerate(vo_list):
        tower_name = mesh_names[tower_id - 1]  
        tower_positions[tower_name] = base_positions[position_idx]
        tower_coords[tower_name] = (base_positions[position_idx][0], base_positions[position_idx][1])

    # Spawn towers at their assigned positions
    for i, tower_name in enumerate(mesh_names):
        x, y, z = tower_positions[tower_name]
        
        mesh_pose = geometry_msgs.msg.PoseStamped()
        mesh_pose.header.frame_id = 'world'
        mesh_pose.pose.position.x = x
        mesh_pose.pose.position.y = y
        mesh_pose.pose.position.z = z

        path_object.add_mesh(tower_name, mesh_pose, MESH_FILE_PATH[i], (.00095, .00095, .00095))
        print(f"Added {tower_name} at position ({x}, {y}, {z})")

def save_voice_command(file_path="/home/cynthia/Robotics/voice_command.txt", language="zh-TW"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("🎤 等待語音輸入中（將於 1 秒後開始錄音）...")
    time.sleep(1)

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("📢 請開始說話...")
        audio = recognizer.listen(source)
        print("🧠 辨識中...")

    try:
        recognized_text = recognizer.recognize_google(audio, language=language)
        print(f"✅ 認識結果：{recognized_text}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(recognized_text + "\n")

        print(f"💾 成功寫入：{file_path}")
        return recognized_text

    except sr.UnknownValueError:
        print("❌ 無法辨識語音")
        return None
    except sr.RequestError as e:
        print(f"❌ 語音辨識請求失敗: {e}")
        return None

def read_voice_commands(path="/home/cynthia/Robotics/voice_command.txt"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ 語音指令讀取失敗: {e}")
        return []

def parse_instruction(instruction):
    """
    支援格式：
    - 從[位置]移到[位置]
    - 從[位置]移動到[位置]
    - 從[位置]搬到[位置]
    - 從[位置]到[位置]
    - 到[位置]
    - 移到[位置]
    - 搬到[位置]
    - 終點位置為[位置]
    - 位置可用 a/A/左、b/B/中、c/C/右 表示
    回傳 tuple: (src, dst)，若只有終點，src 回傳 None
    """
    import re

    pos_map = {
        'a': 'A', 'A': 'A', '左': 'A', '左邊': 'A',
        'b': 'B', 'B': 'B', '中': 'B', '中間': 'B',
        'c': 'C', 'C': 'C', '右': 'C', '右邊': 'C',
    }

    instr = instruction.strip().replace(" ", "")

    # 雙位置形式
    patterns = [
        r"從(.*?)搬到(.*?)",
        r"從(.*?)移動到(.*?)",
        r"從(.*?)移到(.*?)",
        r"從(.*?)到(.*?)",
    ]

    for pat in patterns:
        m = re.match(pat, instr)
        if m:
            src = pos_map.get(m.group(1), None)
            dst = pos_map.get(m.group(2), None)
            if src and dst:
                return src, dst
            else:
                raise ValueError(f"❌ 無法解析位置：{m.group(1)} 或 {m.group(2)}")

    # 單位置形式（只指定目標）
    single_patterns = [
        r"到(.*?)",
        r"移到(.*?)",
        r"搬到(.*?)",
        r"終點位置為(.*?)"
    ]

    for pat in single_patterns:
        m = re.match(pat, instr)
        if m:
            dst = pos_map.get(m.group(1), None)
            if dst:
                return None, dst
            else:
                raise ValueError(f"❌ 無法解析目標位置：{m.group(1)}")

    raise ValueError(f"❌ 無法解析語音指令格式：{instruction}")


def apply_initial_offset(angles):
    for i in range(4):
        angles[i] = angles[i] - initial_offset[i]
    return angles

def send_joint_angles_from_file(file_path, delay=0.05):
    rospy.init_node('txt_to_real_arm_publisher', anonymous=True)
    pub = rospy.Publisher('/real_robot_arm_joint', Float64MultiArray, queue_size=10)
    rospy.sleep(1.0)

    with open(file_path, 'r') as f:
        lines = f.readlines()

    rospy.loginfo(f"📄 開始傳送 {len(lines)} 行關節角度")

    for i, line in enumerate(lines):
        try:
            angles = list(map(float, line.strip().split(',')))
            if len(angles) < 5:
                rospy.logwarn(f"⚠️ 第 {i+1} 行資料不足，略過")
                continue
            
            
            angles = apply_initial_offset(angles)
            msg = Float64MultiArray(data=angles)
            pub.publish(msg)
            rospy.loginfo(f"📤 第 {i+1} 點: {['%.3f' % a for a in angles]}")
            rospy.sleep(delay)
        except ValueError:
            rospy.logwarn(f"⚠️ 第 {i+1} 行格式錯誤，略過")

    rospy.loginfo("✅ 傳送完畢")

def main():
    
    try:
        # Initialize voice command subscriber
        # rospy.Subscriber('/voice_case_cmd', String, voice_callback)
        image_path = capture_image_after_delay()
        if image_path:
            image = cv2.imread(image_path)
            vision_order = detect_color_positions(image)
            vo = vision_order[::-1]
            print(f"📷 Vision 辨識順序為: {vo}")
        spawn_hanoi_towers(path_object,vo)

        voice_result = save_voice_command()
    
        if voice_result is None:
            print("❌ 無法獲得語音輸入")
            return
        
        # 解析語音指令以獲得方向
        try:
            if "左" in voice_result:
                voice_direction = "左"
            elif "右" in voice_result:
                voice_direction = "右"
            elif "中" in voice_result:
                voice_direction = "中"
        except:
            # 如果解析失敗，直接使用原始語音結果
            voice_direction = voice_result
        
        sys.stderr.write("Finish initializing. Press Enter to continue..."); input()    

        try:
            if vo == [3,2,1]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_11_interpolation.txt', delay=delay_time)
            elif vo == [2,3,1]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_12_interpolation.txt', delay=delay_time)
            elif vo == [2,1,3]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_31_interpolation.txt', delay=delay_time)
            elif vo == [3,1,2]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_32_interpolation.txt', delay=delay_time)
            elif vo == [1,2,3]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_51_interpolation.txt', delay=delay_time)
            elif vo == [1,3,2]:
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/ini_52_interpolation.txt', delay=delay_time)
        except rospy.ROSInterruptException:
            print("send angles error")

        index_of_1 = vo.index(1)

        if index_of_1 == 2:
            if voice_direction == '中':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_1.txt', delay=delay_time)
            elif voice_direction == '右':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_2.txt', delay=delay_time)
            elif voice_direction =='左':
                print("~finished~")
            else:
                print(f"index_of_1=0 時，不支援語音指令: {voice_direction}")
        
        elif index_of_1 == 1:
            if voice_direction == '左':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_3.txt', delay=delay_time)
            elif voice_direction == '右':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_4.txt', delay=delay_time)
            elif voice_direction =='中':
                print("~finished~")
            else:
                print(f"index_of_1=1 時，不支援語音指令: {voice_direction}")
        
        elif index_of_1 == 0:
            if voice_direction == '左':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_5.txt', delay=delay_time)
            elif voice_direction == '中':
                send_joint_angles_from_file('/home/cynthia/Robotics/trajectory/case_6.txt', delay=delay_time)
            elif voice_direction =='右':
                print("~finished~")
            else:
                print(f"index_of_1=2 時，不支援語音指令: {voice_direction}")
        
        else:
            print(f"不支援的 index_of_1 值: {index_of_1}")

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == '__main__':
    main()