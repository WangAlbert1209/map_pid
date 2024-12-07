from CPPN.nn import NNGraph
from CPPN.parallel import ParallelEvaluator
from neat1 import NEAT
from utils.config import Config
from utils.image_utils import arr_to_img

max_conn = 100


def generate_image_from_genome(genome, output_path='./best.png'):
    nn = NNGraph.from_genome(genome)

    # 定义图像的形状和输入数据
    img_shape = (50, 50)
    x = np.linspace(-1, 1, img_shape[0])
    y = np.linspace(-1, 1, img_shape[1])
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)

    inputs = np.stack([xx.ravel(), yy.ravel(), r.ravel(), np.ones_like(r.ravel())])
    inputs = inputs.astype(np.float32)

    outputs = nn.eval(inputs)
    outputs = outputs.reshape(img_shape)
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
    outputs = (outputs > 0.5).astype(int)
    # 转换并保存图像
    img_arr = outputs * 255.
    img_arr = img_arr
    img = arr_to_img(img_arr, normalize=False)
    img.save(output_path)
    print(f"Image saved as '{output_path}'")


from PIL import Image
import numpy as np


def get_mean_pattern(pops, output_path='./mean.png'):
    # Initialize parameters for image shape and grid
    img_shape = (50, 50)
    num_genomes = len(pops)

    # Create grid only once
    x = np.linspace(-1, 1, img_shape[0])
    y = np.linspace(-1, 1, img_shape[1])
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    inputs = np.stack([xx.ravel(), yy.ravel(), r.ravel(), np.ones_like(r.ravel())]).astype(np.float32)

    # Accumulate all patterns
    accumulated_pattern = np.zeros(img_shape, dtype=np.float32)

    for genome in pops:
        # Generate a pattern for each genome
        nn = NNGraph.from_genome(genome)

        # Evaluate network and reshape output to image shape
        outputs = nn.eval(inputs)
        outputs = outputs.reshape(img_shape)

        # Accumulate the outputs
        accumulated_pattern += outputs

    # Calculate the mean pattern by dividing by the number of genomes
    mean_pattern = accumulated_pattern / num_genomes

    # Normalize the mean pattern to a 0-255 scale
    mean_pattern = (mean_pattern - mean_pattern.min()) / (mean_pattern.max() - mean_pattern.min() + 1e-8)
    mean_pattern = (mean_pattern * 255).astype(np.uint8)

    # Convert to image and save
    img = Image.fromarray(mean_pattern, 'L')
    img.save(output_path)
    print(f"Mean pattern image saved as '{output_path}'")


# 定义生成圆形图案的目标函数
def square_pattern(x, y):
    # 创建一个与 x 和 y 相同形状的数组，初始值为 0
    pattern = np.zeros_like(x)

    # 条件判断，生成在正方形内部的区域
    pattern[(np.abs(x) <= 0.5) & (np.abs(y) <= 0.5)] = 1

    return pattern


def ellipse_pattern(x, y):
    # 创建一个与 x 和 y 相同形状的数组，初始值为 0
    pattern = np.zeros_like(x)

    # 条件判断，生成在椭圆内部的区域（这里假设半长轴和半短轴均为 0.5）
    pattern[((x / 0.5) ** 2 + (y / 0.25) ** 2) <= 1] = 1

    return pattern


def load_image_and_create_pattern(image_path, grid_shape, threshold=128):
    """
    Load a heart-shaped image, resize it to fit the provided grid, and return a binary pattern array.

    Args:
    - image_path: Path to the image file.
    - grid_shape: Tuple representing the shape of the x and y grid (height, width).
    - threshold: Threshold for binarizing the image (default is 128).

    Returns:
    - pattern: A binary array where 1 indicates the heart shape, resized to the grid's shape.
    """
    # Load the image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize the image to match the grid shape
    image_resized = image.resize((grid_shape[1], grid_shape[0]))  # width x height

    # Convert image to a binary array (0 or 1 based on threshold)
    image_array = np.array(image_resized)
    pattern = (image_array < threshold).astype(np.float32)

    return pattern


def heart_pattern_on_grid(x, y, image_path):
    """
    Creates a binary heart pattern on a given meshgrid using the provided image.

    Args:
    - x, y: 2D arrays representing the grid coordinates.
    - image_path: Path to the heart-shaped image file.

    Returns:
    - pattern: A binary array where 1 indicates the heart shape, resized to the grid's shape.
    """
    grid_shape = x.shape
    pattern = load_image_and_create_pattern(image_path, grid_shape)
    return pattern


# 评估基因组，计算生成图案的误差

img_shape = (50, 50)
x = np.linspace(-1, 1, img_shape[0])
y = np.linspace(-1, 1, img_shape[1])
xx, yy = np.meshgrid(x, y)
r = np.sqrt(xx ** 2 + yy ** 2)
target = heart_pattern_on_grid(xx, yy, "./heart.png")


# img = arr_to_img(target, normalize=False)
# img.show()


def eval_genome(genome):
    nn = NNGraph.from_genome(genome)

    # 按行拼接
    inputs = np.stack([xx.ravel(), yy.ravel(), r.ravel(), np.ones_like(r.ravel())])
    inputs = inputs.astype(np.float64)
    outputs = nn.eval(inputs)
    outputs = outputs.reshape(img_shape)

    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)
    symmetry_x = np.mean(np.abs(outputs - np.flip(outputs, axis=1)))
    symmetry_y = np.mean(np.abs(outputs - np.flip(outputs, axis=0)))
    outputs = (outputs > 0.5).astype(int)
    err = np.sum((outputs - target) ** 2)
    return -err, [symmetry_x, symmetry_y]


def main():
    # 初始化 NEAT
    config = Config.load('./config.json')
    neat = NEAT(config)
    # print(multiprocessing.cpu_count())
    pe = ParallelEvaluator(10, eval_genome)

    init_pops = neat.populate()
    pe.evaluate(list(init_pops.values()))
    for id, g in init_pops.items():
        neat.map_archive.add_to_archive(g)
    # 进化过程
    map_archive = neat.run(pe.evaluate, num_generations=config.gen)
    get_mean_pattern(map_archive.archive.values())

    minf, maxf, best, worst = map_archive.display_archive()
    print(f"min {minf}, max {maxf}")
    print(f'best {best.fitness} b:{best.behavior}')
    print(f'worst {worst.fitness} b:{worst.behavior}')
    generate_image_from_genome(best)
    best.draw()
    worst.draw()


if __name__ == '__main__':
    main()
