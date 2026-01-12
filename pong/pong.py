import pygame
import numpy as np

class Environment:

    def __init__(self, render=False, width=800, height=600, paddle_size=(20, 120), ball_radius=10):
        self.render_mode = render
        self.width = width
        self.height = height
        self.paddle_width, self.paddle_height = paddle_size
        self.ball_radius = ball_radius

        self.paddle_speed = 10
        self.ball_speed = 10

        self.ball_position = np.array([width//2, height//2], dtype=float)
        self.ball_velocity = np.array([self.ball_speed, self.ball_speed], dtype=float)
        self.left_paddle_y = height//2 - self.paddle_height//2
        self.right_paddle_y = height//2 + self.paddle_height//2
        self.done = False

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Pong')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        self.score_left = 0
        self.score_right = 0


    def reset_game(self):
        self.ball_position = np.array([self.width//2, self.height//2], dtype=float)
        self.ball_velocity = np.array([self.ball_speed * np.random.choice([-1, 1]), self.ball_speed * np.random.choice([-1, 1])], dtype=float)
        self.left_paddle_y = self.height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.height // 2 + self.paddle_height // 2
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array([self.ball_position[0] / self.width,
                         self.ball_position[1] / self.height,
                         self.ball_velocity[0] / self.ball_speed,
                         self.ball_velocity[1] / self.ball_speed,
                         self.left_paddle_y / self.height,
                         self.right_paddle_y / self.height], dtype=float)

    def step(self, action_left, action_right):
        self.left_paddle_y += action_left * self.paddle_speed
        self.right_paddle_y += action_right * self.paddle_speed

        self.left_paddle_y = np.clip(self.left_paddle_y, 0, self.height - self.paddle_height)
        self.right_paddle_y = np.clip(self.right_paddle_y, 0, self.height - self.paddle_height)

        self.ball_position += self.ball_velocity


        # hit ceiling or floor
        if self.ball_position[1] <= self.ball_radius or self.ball_position[1] >= self.height - self.ball_radius:
            self.ball_velocity[1] *= -1

        reward_left = 0
        reward_right = 0

        # collision with left paddle
        if (self.ball_position[0] - self.ball_radius <= self.paddle_width and
            self.left_paddle_y <= self.ball_position[1] <= self.left_paddle_y + self.paddle_height):
            self.ball_velocity[0] *= -1
            reward_left = 1

        # collision with right paddle
        if (self.ball_position[0] - self.ball_radius <= self.paddle_width and
            self.right_paddle_y <= self.ball_position[1] <= self.right_paddle_y + self.paddle_height):
            self.ball_velocity[0] *= -1
            reward_right = 1

        # right score
        if self.ball_position[0] < 0:
            self.score_right += 1
            reward_right += 100
            reward_left -= 100
            self.done = True

        # left score
        if self.ball_position[0] > self.width:
            self.score_left += 1
            reward_left += 100
            reward_right -= 100
            self.done = True

        if self.render_mode:
            self.render()

        return self.get_state(), reward_left, reward_right, self.done


    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255,255,255), (0, self.left_paddle_y, self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.width - self.paddle_width, self.right_paddle_y, self.paddle_width, self.paddle_height))

        pygame.draw.circle(self.screen, (255,255,255), self.ball_position.astype(int), self.ball_radius)

        score = self.font.render(f"{self.score_left} : {self.score_right}", True, (255,255,255))
        self.screen.blit(score, (self.width // 2 - 20, 10))
        pygame.display.flip()
        self.clock.tick(60)

def test():
    env = Environment(render=True)
    state = env.reset_game()

    done = False

    while not done:
        action_left = np.random.choice([-1, 0, 1])
        action_right = np.random.choice([-1, 0, 1])
        state, r_left, r_right, done = env.step(action_left, action_right)
