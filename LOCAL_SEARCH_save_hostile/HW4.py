import pygame
import random
import math
from datetime import datetime

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
TILE_SIZE = 40
ROWS, COLS = HEIGHT // TILE_SIZE, WIDTH // TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rescue the Hostage - Local Search")

# Colors
WHITE = (240, 248, 255)
RED = (255, 69, 0)      # Hostage color
BLUE = (30, 144, 255)   # Player color
LIGHT_GREY = (211, 211, 211) # Background grid color
FLASH_COLOR = (50, 205, 50) # Victory flash color
BUTTON_COLOR = (50, 205, 50) # Button color
BUTTON_TEXT_COLOR = (255, 255, 255) # Button text color

# Load images for player, hostage, and walls
player_image = pygame.image.load("images/AI1.png")  
hostage_image = pygame.image.load("images/AI2.png")  
wall_images = [
    pygame.image.load("images/AI3.png"),
    pygame.image.load("images/AI4.png"),
    pygame.image.load("images/AI5.png")
]

# Resize images to fit the grid
wall_images = [pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE)) for img in wall_images]
player_image = pygame.transform.scale(player_image, (TILE_SIZE, TILE_SIZE))
hostage_image = pygame.transform.scale(hostage_image, (TILE_SIZE, TILE_SIZE))

# Constants for recent positions
MAX_RECENT_POSITIONS = 10
GENERATION_LIMIT = 50
MUTATION_RATE = 0.1

# Function to generate obstacles
def generate_obstacles(num_obstacles):
    obstacles = []
    while len(obstacles) < num_obstacles:
        new_obstacle = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        if new_obstacle not in obstacles:  # Make sure obstacles are not overlapping
            obstacles.append(new_obstacle)
    obstacle_images = [random.choice(wall_images) for _ in obstacles]
    return obstacles, obstacle_images

# Function to start a new game
def start_new_game():
    global player_pos, hostage_pos, recent_positions, obstacles, obstacle_images, temperature, iter, full_path, steps, start, obstacle_meets_cnt
    obstacles, obstacle_images = generate_obstacles(30)
    recent_positions = []
    temperature = 100
    # first element in dfs path is the source so we start from 1 index
    iter = 1
    full_path = []
    steps = 0
    start = datetime.now()
    obstacle_meets_cnt = 0

    # Generate player and hostage positions with a larger distance
    while True:
        player_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        hostage_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        distance = math.dist(player_pos, hostage_pos)
        if distance > 8 and player_pos not in obstacles and hostage_pos not in obstacles:
            break

# Function to get manhatan distance of two cell in grid
def get_manhatan_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def is_valid_player_pos(player_pos, obstacles):
    return (
        player_pos[0] >= 0 
        and player_pos[0] < COLS 
        and player_pos[1] >= 0 
        and player_pos[1] < ROWS 
        and player_pos not in obstacles
    )

# Function to move the player closer to the hostage using Hill Climbing algorithm
def hill_climbing(player, hostage, obstacles):
    move_x = [1, -1, 0, 0]
    move_y = [0, 0, 1, -1]
    min_distance = get_manhatan_distance(player, hostage)
    min_pos = player
    for i in range(4):
        neighbor = [player[0]+move_x[i], player[1]+move_y[i]]
        if is_valid_player_pos(neighbor, obstacles):
            new_distance = get_manhatan_distance(neighbor, hostage)
            if new_distance < min_distance:
                min_distance = new_distance
                min_pos = neighbor
    return min_pos

 

# Function for Simulated Annealing
def simulated_annealing(player, hostage, obstacles):
    global temperature  # Initial temperature
    cooling_rate = 0.99
    min_distance = get_manhatan_distance(player, hostage)
    min_pos = player
    
    def anneal(cur_player):
        global temperature
        nonlocal min_distance
        move = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(move)

        cur_distance = get_manhatan_distance(cur_player, hostage)
        cur_pos = cur_player
        bad_moves = []
        for i in range(4):
            neighbor = [player[0]+move[i][0], player[1]+move[i][1]]
            if not is_valid_player_pos(neighbor, obstacles):
                continue
            new_distance = get_manhatan_distance(neighbor, hostage)
            if new_distance < cur_distance:
                if new_distance < min_distance:
                    min_distance = new_distance
                    min_pos = neighbor
                return neighbor
            else:
                bad_moves.append(i)
        
        
        for i in bad_moves:
            neighbor = [player[0]+move[i][0], player[1]+move[i][1]]
            if random.random() < acceptance_probability(cur_distance, new_distance, temperature):
                    return neighbor
        
        return cur_pos  

    # Acceptance probability function
    def acceptance_probability(old_cost, new_cost, temp):
        probability = math.exp(-abs(new_cost - old_cost) / temp)
        return probability
    
    new_pos = anneal(player)
    temperature *= cooling_rate
    return new_pos
    

# Function for Genetic Algorithm
def genetic_algorithm(player, hostage, obstacles):
    population_size = 20
    generations_size = 50
    generations = []
    mutation_chacne = 0.05

    marked = dict()
    path = list()

    def dfs(src, dest, par=-1):
        if marked.get(src) == True:
            return False
        
        marked[src] = True

        if src == dest:
            path.append(src)
            return True
        

        move = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(move)
        for i in range(4):
            new_cell = (src[0]+move[i][0], src[1]+move[i][1])
            if not is_valid_player_pos(list(new_cell), obstacles):
                continue
            if dfs(new_cell, dest, src):
                path.append(src)
                return True
        
        return False
        
    # Fitness function
    def fitness(individual):
        return len(individual)

    # Generate random population
    def generate_population():
        for i in range(population_size):
            nonlocal marked, path
            marked = dict()
            path = list()
            dfs(tuple(player), tuple(hostage))
            generations.append(path)

    # Crossover function
    def crossover(parent1, parent2):
        # in common cells in path of parent1 and parent2
        chrom_pairs = list()
        for i in range(len(parent1)):
            if parent1[i] in parent2:
                chrom_pairs.append((i, parent2.index(parent1[i])))

        idx1, idx2 = random.choice(chrom_pairs)
        
        child1 = parent1[:idx1+1] + parent2[idx2+1:]
        if random.random() < mutation_chacne:
            child1 = mutate(child1)
        child2 = parent2[:idx2+1] + parent1[idx1+1:]
        if random.random() < mutation_chacne:
            child2 = mutate(child2)
        return [child1, child2]

    # Mutation function
    def mutate(individual):
        nonlocal marked, path
        marked = dict()
        path = list()

        gen_len = len(individual)
        mut_len = random.choice(range(2, min(gen_len, 10)))
        mut_pos = random.choice(range(gen_len-mut_len))

        dfs(individual[mut_pos], individual[mut_pos+mut_len])
        path.reverse()
        new_gen = individual[:mut_pos] + path + individual[mut_pos+mut_len+1:]
        return new_gen

    generate_population()
    for _ in range(generations_size):
        new_generations = list()
        for par1, par2 in zip(generations, generations[1:]+[generations[0]], strict=True):
            children = crossover(par1, par2)
            new_generations += children
        new_generations.sort(key=lambda gen: fitness(gen))
        generations = new_generations[:population_size]
    
    return generations[0]

#Objective: Check if the player is stuck in a repeating loop.
def in_loop(recent_positions, player):
    return player in recent_positions
    

#Objective: Make a random safe move to escape loops or being stuck.
def random_move(player, obstacles):
    while True:
        new_player_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        if new_player_pos != player and new_player_pos not in obstacles:
            break
    return new_player_pos

#Objective: Update the list of recent positions. 
def store_recent_position(recent_positions, new_player_pos, max_positions=MAX_RECENT_POSITIONS):
    if len(recent_positions) >= max_positions:
        recent_positions.pop(0)
    recent_positions.append(new_player_pos)

# Function to show victory flash
def victory_flash():
    for _ in range(5):
        screen.fill(FLASH_COLOR)
        pygame.display.flip()
        pygame.time.delay(100)
        screen.fill(WHITE)
        pygame.display.flip()
        pygame.time.delay(100)

# Function to show a button and wait for player's input
def show_button_and_wait(message, button_rect):
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, BUTTON_TEXT_COLOR)
    button_rect.width = text.get_width() + 20
    button_rect.height = text.get_height() + 10
    button_rect.center = (WIDTH // 2, HEIGHT // 2)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    screen.blit(text, (button_rect.x + (button_rect.width - text.get_width()) // 2,
                       button_rect.y + (button_rect.height - text.get_height()) // 2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False

# Function to get the algorithm choice from the player
def get_algorithm_choice():
    print("Choose an algorithm:")
    print("1: Hill Climbing")
    print("2: Simulated Annealing")
    print("3: Genetic Algorithm")

    while True:
        choice = input("Enter the number of the algorithm you want to use (1/2/3): ")
        if choice == "1":
            return hill_climbing
        elif choice == "2":
            return simulated_annealing
        elif choice == "3":
            return genetic_algorithm
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")




runs = []
exec_times = []
obstacle_meets = []
# Main game loop
running = True
clock = pygame.time.Clock()
start_new_game()
button_rect = pygame.Rect(0, 0, 0, 0)

# Get the algorithm choice from the player
chosen_algorithm = get_algorithm_choice()

while running:
    steps += 1
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform the chosen algorithm step
    if chosen_algorithm is genetic_algorithm:
        # in first step of genetic get full path
        if not full_path:
            full_path = chosen_algorithm(player_pos, hostage_pos, obstacles)
            full_path.reverse()
        new_player_pos = list(full_path[iter])
        iter += 1
    else:
        new_player_pos = chosen_algorithm(player_pos, hostage_pos, obstacles)

    # Check for stuck situations
    if new_player_pos == player_pos or in_loop(recent_positions, new_player_pos):
        # Perform a random move when stuck
        if new_player_pos == player_pos:
            obstacle_meets_cnt += 1
        new_player_pos = random_move(player_pos, obstacles)

    # Update recent positions
    store_recent_position(recent_positions, new_player_pos)
    # Update player's position
    player_pos = new_player_pos

    # Draw the grid background
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, LIGHT_GREY, rect, 1)

    # Draw obstacles
    for idx, obs in enumerate(obstacles):
        obs_rect = pygame.Rect(obs[0] * TILE_SIZE, obs[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(obstacle_images[idx], obs_rect)

    # Draw player
    player_rect = pygame.Rect(player_pos[0] * TILE_SIZE, player_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    screen.blit(player_image, player_rect)

    # Draw hostage
    hostage_rect = pygame.Rect(hostage_pos[0] * TILE_SIZE, hostage_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    screen.blit(hostage_image, hostage_rect)

    # Check if player reached the hostage
    if player_pos == hostage_pos:
        print(f"Hostage Rescued!\t total steps: {steps}")
        runs.append(steps)
        exec_times.append((datetime.now() - start).total_seconds())
        obstacle_meets.append(obstacle_meets_cnt)
        print(f"Iteration: {len(runs)}")
        print(f"Average steps up to now: {sum(runs)/len(runs)}")
        print(f"Average execution time up to now: {sum(exec_times)/len(exec_times)}")
        print(f"Average obstacle meets up to now: {sum(obstacle_meets)/len(obstacle_meets)}")
        victory_flash()  # Show the victory flash
        # show_button_and_wait("New Game", button_rect)
        start_new_game()

    # Update the display
    pygame.display.flip()
    clock.tick(5)  # Lower frame rate for smoother performance