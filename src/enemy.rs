use cgmath::{Point3, Vector3, InnerSpace};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Enemy {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub health: f32,
    pub max_health: f32,
    pub speed: f32,
    pub attack_range: f32,
    pub attack_damage: f32,
    pub alive: bool,
    state: EnemyState,
    state_timer: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EnemyState {
    Idle,
    Wandering,
    Chasing,
    Attacking,
}

impl Enemy {
    pub fn new(position: Point3<f32>) -> Self {
        Self {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            health: 20.0,
            max_health: 20.0,
            speed: 3.0,
            attack_range: 2.0,
            attack_damage: 5.0,
            alive: true,
            state: EnemyState::Idle,
            state_timer: 0.0,
        }
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>) {
        if !self.alive {
            return;
        }

        self.state_timer -= dt;

        let distance_to_player = (player_pos - self.position).magnitude();

        // State transitions
        if distance_to_player < 15.0 {
            self.state = EnemyState::Chasing;
        } else if distance_to_player < self.attack_range {
            self.state = EnemyState::Attacking;
        } else if self.state_timer <= 0.0 {
            self.state = if rand::random::<f32>() > 0.5 {
                EnemyState::Wandering
            } else {
                EnemyState::Idle
            };
            self.state_timer = rand::thread_rng().gen_range(2.0..5.0);
        }

        // Behavior based on state
        match self.state {
            EnemyState::Idle => {
                self.velocity = Vector3::new(0.0, 0.0, 0.0);
            }
            EnemyState::Wandering => {
                if self.state_timer <= 0.0 {
                    let angle = rand::thread_rng().gen_range(0.0..std::f32::consts::TAU);
                    self.velocity = Vector3::new(angle.cos(), 0.0, angle.sin()) * self.speed;
                }
            }
            EnemyState::Chasing => {
                let direction = (player_pos - self.position).normalize();
                self.velocity = Vector3::new(direction.x, 0.0, direction.z) * self.speed;
            }
            EnemyState::Attacking => {
                self.velocity = Vector3::new(0.0, 0.0, 0.0);
            }
        }

        // Apply velocity
        self.position += self.velocity * dt;

        // Simple gravity
        self.position.y -= 9.8 * dt;
        if self.position.y < 30.0 {
            self.position.y = 30.0;
        }
    }

    pub fn take_damage(&mut self, damage: f32) {
        self.health -= damage;
        if self.health <= 0.0 {
            self.alive = false;
        }
    }

    pub fn get_color(&self) -> [f32; 3] {
        if self.state == EnemyState::Attacking {
            [1.0, 0.0, 0.0] // Red when attacking
        } else if self.state == EnemyState::Chasing {
            [1.0, 0.5, 0.0] // Orange when chasing
        } else {
            [0.5, 0.0, 0.5] // Purple otherwise
        }
    }
}

pub struct EnemyManager {
    pub enemies: Vec<Enemy>,
    spawn_timer: f32,
    spawn_interval: f32,
    max_enemies: usize,
}

impl EnemyManager {
    pub fn new(spawn_interval: f32, max_enemies: usize) -> Self {
        Self {
            enemies: Vec::new(),
            spawn_timer: 0.0,
            spawn_interval,
            max_enemies,
        }
    }

    pub fn update(&mut self, dt: f32, player_pos: Point3<f32>) {
        // Update existing enemies
        for enemy in &mut self.enemies {
            enemy.update(dt, player_pos);
        }

        // Remove dead enemies
        self.enemies.retain(|e| e.alive);

        // Spawn new enemies
        self.spawn_timer += dt;
        if self.spawn_timer >= self.spawn_interval && self.enemies.len() < self.max_enemies {
            self.spawn_enemy(player_pos);
            self.spawn_timer = 0.0;
        }
    }

    fn spawn_enemy(&mut self, player_pos: Point3<f32>) {
        let mut rng = rand::thread_rng();

        // Spawn enemies around the player but not too close
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let distance = rng.gen_range(20.0..40.0);

        let spawn_pos = Point3::new(
            player_pos.x + angle.cos() * distance,
            30.0,
            player_pos.z + angle.sin() * distance,
        );

        self.enemies.push(Enemy::new(spawn_pos));
    }

    pub fn check_player_damage(&self, player_pos: Point3<f32>) -> f32 {
        let mut total_damage = 0.0;
        for enemy in &self.enemies {
            if enemy.alive && enemy.state == EnemyState::Attacking {
                let distance = (player_pos - enemy.position).magnitude();
                if distance < enemy.attack_range {
                    total_damage += enemy.attack_damage;
                }
            }
        }
        total_damage
    }
}
