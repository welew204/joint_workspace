/**
 * openlayers/src/ol/math.js
 *
 * Converts degrees to radians.
 *
 * @param {number} angleInDegrees Angle in degrees.
 * @return {number} Angle in radians.
 */
export function toRadians(angleInDegrees) {
  return (angleInDegrees * Math.PI) / 180;
}
/**
 * OpenLayers is the library it's from
 * openlayers/src/ol/sphere.js
 *
 * Returns the spherical area for a list of coordinates.
 *
 * [Reference](https://trs.jpl.nasa.gov/handle/2014/40409)
 * Robert. G. Chamberlain and William H. Duquette, "Some Algorithms for
 * Polygons on a Sphere", JPL Publication 07-03, Jet Propulsion
 * Laboratory, Pasadena, CA, June 2007
 *
 * @param {Array<import("./coordinate.js").Coordinate>} coordinates List of coordinates of a linear
 * ring. If the ring is oriented clockwise, the area will be positive,
 * otherwise it will be negative.
 * @param {number} radius The sphere radius.
 * @return {number} Area (in square meters).
 */
function getAreaInternal(coordinates, radius) {
  let area = 0;
  const len = coordinates.length;
  let x1 = coordinates[len - 1][0];
  let y1 = coordinates[len - 1][1];
  for (let i = 0; i < len; i++) {
    const x2 = coordinates[i][0];
    const y2 = coordinates[i][1];
    area +=
      toRadians(x2 - x1) *
      (2 + Math.sin(toRadians(y1)) + Math.sin(toRadians(y2)));
    x1 = x2;
    y1 = y2;
  }
  return (area * radius * radius) / 2.0;
}
