import pygame
import random

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 450
FPS = 60

# Colors
WHITE = (255, 255, 255)

# Load images (placeholders for now, replace with pixel art assets)
background = pygame.image.load("images/bg.jpg")
slingshot = pygame.image.load("images/sling.png")
bird = pygame.image.load("images/bird.png")
pig = pygame.image.load("images/pig.png")
wood_block = pygame.image.load("images/block.png")

# Scale images for pixelated look
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
slingshot = pygame.transform.scale(slingshot, (60, 120))

# Game screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pixel Angry Birds")


# Classes
class Bird:
    def __init__(self, x, y):
        self.image = pygame.transform.scale(bird, (40, 40))
        self.x, self.y = x, y
        self.velocity = [0, 0]
        self.launched = False

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

    def launch(self, power_x, power_y):
        self.velocity = [power_x, power_y]
        self.launched = True

    def update(self):
        if self.launched:
            self.x += self.velocity[0]
            self.y += self.velocity[1]
            self.velocity[1] += 0.5  # Gravity effect


class Pig:
    def __init__(self, x, y):
        self.image = pygame.transform.scale(pig, (40, 40))
        self.x, self.y = x, y

    def draw(self):
        screen.blit(self.image, (self.x, self.y))


# Initialize objects
bird_obj = Bird(100, 270)
pigs = [Pig(random.randint(500, 700), random.randint(300, 350))]

# Main loop
running = True
while running:
    screen.blit(background, (0, 0))
    screen.blit(slingshot, (78, 270))
    bird_obj.draw()
    bird_obj.update()

    for pig in pigs:
        pig.draw()

    pygame.display.update()
    pygame.time.Clock().tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not bird_obj.launched:
            bird_obj.launch(7, -10)

pygame.quit()
