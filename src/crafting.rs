use crate::block::BlockType;
use crate::inventory::Inventory;

#[derive(Debug, Clone)]
pub struct Recipe {
    pub inputs: Vec<(BlockType, f32)>,
    pub output: (BlockType, f32),
}

pub struct CraftingSystem {
    recipes: Vec<Recipe>,
}

impl CraftingSystem {
    pub fn new() -> Self {
        let mut system = Self {
            recipes: Vec::new(),
        };
        system.register_default_recipes();
        system
    }

    fn register_default_recipes(&mut self) {
        // Wood -> Planks (1 wood = 4 planks)
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Wood, 1.0)],
            output: (BlockType::Planks, 4.0),
        });

        // Planks -> Wood (4 planks = 1 wood)
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Planks, 4.0)],
            output: (BlockType::Wood, 1.0),
        });

        // Stone -> Cobblestone
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Stone, 1.0)],
            output: (BlockType::Cobblestone, 1.0),
        });

        // Cobblestone -> Stone
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Cobblestone, 1.0)],
            output: (BlockType::Stone, 1.0),
        });

        // Dirt + Grass -> More Grass
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Dirt, 2.0), (BlockType::Grass, 1.0)],
            output: (BlockType::Grass, 3.0),
        });

        // Sand -> Stone (smelting)
        self.recipes.push(Recipe {
            inputs: vec![(BlockType::Sand, 4.0)],
            output: (BlockType::Stone, 1.0),
        });
    }

    pub fn can_craft(&self, inventory: &Inventory, recipe_index: usize) -> bool {
        if recipe_index >= self.recipes.len() {
            return false;
        }

        let recipe = &self.recipes[recipe_index];
        self.has_ingredients(inventory, &recipe.inputs)
    }

    fn has_ingredients(&self, inventory: &Inventory, ingredients: &[(BlockType, f32)]) -> bool {
        for &(block_type, required_count) in ingredients {
            let mut total = 0.0;
            for slot in &inventory.slots {
                if let Some(stack) = slot {
                    if stack.block_type == block_type {
                        total += stack.count;
                    }
                }
            }
            if total < required_count {
                return false;
            }
        }
        true
    }

    pub fn craft(&self, inventory: &mut Inventory, recipe_index: usize) -> bool {
        if !self.can_craft(inventory, recipe_index) {
            return false;
        }

        let recipe = &self.recipes[recipe_index];

        // Remove ingredients
        for &(block_type, required_count) in &recipe.inputs {
            let mut remaining = required_count;
            for i in 0..inventory.size {
                if remaining <= 0.0 {
                    break;
                }
                if let Some(stack) = &inventory.slots[i] {
                    if stack.block_type == block_type {
                        let to_remove = remaining.min(stack.count);
                        inventory.remove_item(i, to_remove);
                        remaining -= to_remove;
                    }
                }
            }
        }

        // Add output
        inventory.add_item(recipe.output.0, recipe.output.1)
    }

    pub fn get_recipes(&self) -> &[Recipe] {
        &self.recipes
    }

    pub fn get_available_recipes(&self, inventory: &Inventory) -> Vec<usize> {
        self.recipes
            .iter()
            .enumerate()
            .filter(|(i, _)| self.can_craft(inventory, *i))
            .map(|(i, _)| i)
            .collect()
    }
}
