import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from datetime import datetime

def process_cifar10_to_c_array(dataset, index, class_names):
    img_pil, label = dataset[index]

    # img_gray = img_pil.convert('L')

    mean_rgb = [0.4914, 0.4822, 0.4465]
    std_rgb = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((np.mean(mean_rgb),), (np.mean(std_rgb),)) 
    ])
    img_standardized = transform(img_pil)

    img_numpy = img_standardized.numpy()

    abs_max = np.maximum(np.abs(img_numpy).max(), 1e-5)
    scale = 127.0 / abs_max
    img_quantized = np.round(img_numpy * scale)
    img_quantized = np.clip(img_quantized, -128, 127).astype(np.int8)


    label_name = class_names[label]
    array_name = f"cifar10_data_{index}"
    label_name_var = f"cifar10_label_{index}"
    

    c_code = f"// samples {index}: {label_name}\n"
    c_code += f"int8_t {array_name}[3072] = {{\n    "
    for i, val in enumerate(img_quantized.flatten()):

        hex_val = f"0x{val & 0xFF:02x}"
        c_code += f"{hex_val}, "
        if (i + 1) % 16 == 0:
            c_code += "\n    "
    c_code = c_code.strip().rstrip(',') + "\n};\n"
    c_code += f"uint8_t {label_name_var} = {label};\n"
    
    return c_code

def generate_and_save(num_samples, output_filename):

    
    print(f"Loading CIFAR10 data...")
    try:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    except Exception as e:
        print(f"network error: {e}")
        return

    class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"gen {num_samples} test datas to '{output_filename}'...")

    with open(output_filename, 'w') as f:
        f.write("/**\n")
        f.write(f" * @file {output_filename}\n")
        f.write(f" * @brief CIFAR-10 Test Data (Converted to 32x32 Grayscale, Quantized to int8_t)\n")
        f.write(f" * @generated_by generate_cifar10_data.py\n")
        f.write(f" * @date {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(" */\n\n")
        f.write("#ifndef CIFAR10_TEST_DATA_H\n")
        f.write("#define CIFAR10_TEST_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        f.write("// CIFAR-10 Class Labels\n")
        for i, name in enumerate(class_names):
            f.write(f"// {i}: {name}\n")
        f.write("\n")

        for i in range(num_samples):
            print(f" make sample {i+1}/{num_samples}...")
            c_array_code = process_cifar10_to_c_array(testset, i, class_names)
            f.write(c_array_code)

        f.write("#endif // CIFAR10_TEST_DATA_H\n")

    print(f"\nsuccessful！ save to '{output_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate C header file with CIFAR-10 test data.")
    parser.add_argument(
        '-n', '--num_samples', 
        type=int, 
        default=10, 
        help="Number of samples to generate (default: 10)"
    )
    parser.add_argument(
        '-o', '--output_file', 
        type=str, 
        default="BitNetMCU_CIFAR10_test_data.h", 
        help="Output header file name (default: BitNetMCU_CIFAR10_test_data.h)"
    )
    args = parser.parse_args()

    generate_and_save(args.num_samples, args.output_file)