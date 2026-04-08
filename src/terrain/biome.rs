// Biome types for terrain generation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiomeType {
    Desert,
    Forest,
    Mountains,
    Arctic,
    Ocean,
    Plains,
}

// Biome weights for smooth blending between biomes
#[derive(Clone, Copy, Debug)]
pub struct BiomeWeights {
    pub desert: f64,
    pub forest: f64,
    pub mountains: f64,
    pub arctic: f64,
    pub ocean: f64,
    pub plains: f64,
}

impl BiomeWeights {
    pub fn new() -> Self {
        Self {
            desert: 0.0,
            forest: 0.0,
            mountains: 0.0,
            arctic: 0.0,
            ocean: 0.0,
            plains: 0.0,
        }
    }

    pub fn normalize(&mut self) {
        let total = self.desert + self.forest + self.mountains + self.arctic + self.ocean + self.plains;
        if total > 0.0 {
            self.desert /= total;
            self.forest /= total;
            self.mountains /= total;
            self.arctic /= total;
            self.ocean /= total;
            self.plains /= total;
        }
    }

    pub fn dominant(&self) -> BiomeType {
        let mut max_weight = self.desert;
        let mut dominant = BiomeType::Desert;

        if self.forest > max_weight {
            max_weight = self.forest;
            dominant = BiomeType::Forest;
        }
        if self.mountains > max_weight {
            max_weight = self.mountains;
            dominant = BiomeType::Mountains;
        }
        if self.arctic > max_weight {
            max_weight = self.arctic;
            dominant = BiomeType::Arctic;
        }
        if self.ocean > max_weight {
            max_weight = self.ocean;
            dominant = BiomeType::Ocean;
        }
        if self.plains > max_weight {
            dominant = BiomeType::Plains;
        }
        dominant
    }
}
