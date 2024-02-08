import subprocess
import re

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)
        return None

def extract_program_duration(output):
    # Define a regular expression pattern to extract the program duration
    pattern = r"Program duration: (\d+\.?\d*) s"
    match = re.search(pattern, output)
    if match:
        return match.group(0)
    else:
        return None

if __name__ == "__main__":
    # Replace the command with the actual command you want to run
    host = "10.15.40.161"
    threshold = 0.3
    resw = 640
    resh = 480
    
    # data = ["mobilenetv1", "mobilenetv2", "mobiledet", "efficientdet0", "efficientdet1", "efficientdet2", "efficientdet3"]
    data = ["mobilenetv1", "mobilenetv2", "mobiledet"]
    source = ["rtsp://KCKS:majuteru5@10.15.40.48:554/Streaming/Channels/1101"]

    # Run the command multiple times
    for model in data:
        for url in rtsp:
            print(f"Running program - {model}/{url[-3:]}")
            base_command = f"python3 detect.py -m {model} -i {source} --host {host} -t {threshold} -resw {resw} -resh {resh} --debug"

            # Run the command to execute the script
            output = run_command(base_command)

            if output:
                # Extract program duration
                duration_line = extract_program_duration(output)
                if duration_line is not None:
                    print(f"{model} {duration_line}")

                    # Save program duration to a text file
                    with open("program_durations.txt", "a") as file:
                        file.write(model + duration_line + "\n")
                        print(f"Program duration {model} - {url[-3:]} saved to 'program_durations.txt'")
                else:
                    print("Unable to extract program duration.")
            else:
                print("Error running the command.")
