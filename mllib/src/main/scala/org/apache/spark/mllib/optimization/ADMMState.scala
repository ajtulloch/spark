/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.optimization

import breeze.linalg.DenseVector
import breeze.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * The state kept on each partition - the data points, and the x,
 * y, u vectors at each iteration.
 */
case class ADMMState(
  points: Array[LabeledPoint],
  x: DenseVector[Double],
  z: DenseVector[Double],
  u: DenseVector[Double],
  // dual variables used by some algorithms
  dual: Option[DenseVector[Double]])

object ADMMState {
  def apply(points: Iterable[LabeledPoint], initialWeights: Array[Double]): ADMMState = {
    new ADMMState(
      points = points.toArray,
      x = DenseVector(initialWeights),
      z = zeroes(initialWeights.length),
      u = zeroes(initialWeights.length),
      dual = None
    )
  }

  def zeroes(n: Int) = {
    DenseVector(Array.fill(n){ 0.0 })
  }
}
