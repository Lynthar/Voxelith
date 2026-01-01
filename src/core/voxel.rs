//! Voxel data representation.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Material identifier for a voxel.
/// 0 = Air (empty), 1+ = solid materials
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Material(pub u16);

impl Material {
    pub const AIR: Self = Self(0);

    #[inline]
    pub fn is_air(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn is_solid(self) -> bool {
        self.0 != 0
    }
}

/// A single voxel with material and color information.
///
/// Memory layout is optimized for cache efficiency (8 bytes total).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct Voxel {
    /// Material type (0 = air)
    pub material: u16,
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
    /// Alpha component (0-255, used for transparency effects)
    pub a: u8,
    /// Additional flags for special properties
    /// Bit 0: emissive
    /// Bit 1: metallic
    /// Bit 2-7: reserved
    pub flags: u8,
    /// Reserved for future use (e.g., rotation, variant)
    pub _reserved: u8,
}

impl Voxel {
    /// Air voxel (empty space)
    pub const AIR: Self = Self {
        material: 0,
        r: 0,
        g: 0,
        b: 0,
        a: 0,
        flags: 0,
        _reserved: 0,
    };

    /// Create a new solid voxel with the given material and color
    #[inline]
    pub fn new(material: u16, r: u8, g: u8, b: u8) -> Self {
        Self {
            material,
            r,
            g,
            b,
            a: 255,
            flags: 0,
            _reserved: 0,
        }
    }

    /// Create a voxel from RGB color with default material (1)
    #[inline]
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(1, r, g, b)
    }

    /// Create a voxel from RGBA color
    #[inline]
    pub fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self {
            material: 1,
            r,
            g,
            b,
            a,
            flags: 0,
            _reserved: 0,
        }
    }

    /// Check if this voxel is air (empty)
    #[inline]
    pub fn is_air(&self) -> bool {
        self.material == 0
    }

    /// Check if this voxel is solid (not air)
    #[inline]
    pub fn is_solid(&self) -> bool {
        self.material != 0
    }

    /// Get color as [r, g, b, a] array
    #[inline]
    pub fn color(&self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Get color as normalized [0.0-1.0] floats
    #[inline]
    pub fn color_f32(&self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            self.a as f32 / 255.0,
        ]
    }

    /// Check if voxel is emissive
    #[inline]
    pub fn is_emissive(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Set emissive flag
    #[inline]
    pub fn set_emissive(&mut self, emissive: bool) {
        if emissive {
            self.flags |= 0x01;
        } else {
            self.flags &= !0x01;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_size() {
        assert_eq!(std::mem::size_of::<Voxel>(), 8);
    }

    #[test]
    fn test_air_detection() {
        assert!(Voxel::AIR.is_air());
        assert!(!Voxel::AIR.is_solid());

        let solid = Voxel::from_rgb(255, 0, 0);
        assert!(!solid.is_air());
        assert!(solid.is_solid());
    }

    #[test]
    fn test_color_conversion() {
        let v = Voxel::from_rgba(128, 64, 32, 255);
        let c = v.color_f32();
        assert!((c[0] - 0.502).abs() < 0.01);
        assert!((c[1] - 0.251).abs() < 0.01);
        assert!((c[2] - 0.125).abs() < 0.01);
    }
}
