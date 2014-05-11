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

package org.apache.spark.mllib.classification

import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.optimization.{ADMMOptimizer, ADMMState, ADMMUpdater}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD

/*
 * Some helper methods for converting between ScalaNLP's DenseVector
 * and Spark's Vector representation.
 */

case class SparseLogisticRegressionADMMUpdater(
  lambda: Double,
  rho: Double,
  lbfgsMaxNumIterations: Int = 5,
  lbfgsHistory: Int = 10,
  lbfgsTolerance: Double = 1E-4) extends ADMMUpdater {

  import SparseLogisticRegressionADMMUpdater._

  def xUpdate(state: ADMMState): ADMMState = {
      // Our convex objective function that we seek to optimize
    val f = new DiffFunction[DenseVector[Double]] {
      def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val vx = Vectors.fromBreeze(x)
        (objective(state)(vx), gradient(state)(vx).toBreeze.toDenseVector)
      }
    }

    // TODO(tulloch) - it would be nice to have relative tolerance and
    // absolute tolerance here.
    val lbfgs = new LBFGS[DenseVector[Double]](
      maxIter = lbfgsMaxNumIterations,
      m = lbfgsHistory,
      tolerance = lbfgsTolerance)

    val xNew = lbfgs.minimize(f, state.x) // this is the "warm start" approach
    state.copy(x = xNew)
  }

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] = {
    ADMMUpdater.linearZUpdate(lambda = lambda, rho = rho)(states)
  }

  def objective(state: ADMMState)(weights: Vector): Double = {
    val lossObjective = state.points
      .map(lp => {
        val margin = lp.label * (weights.toBreeze dot lp.features.toBreeze)
        -logPhi(margin)
      })
      .sum

    val xxx = (weights.toBreeze - state.z + state.u).toDenseVector // FIXME
    val regularizerObjective = math.pow(breeze.linalg.norm(xxx), 2)
    val totalObjective = lossObjective + rho / 2 * regularizerObjective
    totalObjective
  }

  def gradient(state: ADMMState)(weights: Vector): Vector = {
    val lossGradient = state.points
      .map(lp => {
        val margin = lp.label * (weights.toBreeze dot lp.features.toBreeze)
        lp.features.toBreeze * (lp.label * (phi(margin) - 1))
      })
      .reduce(_ + _)

    val regularizerGradient = (weights.toBreeze - state.z + state.u) * 2.0
    val totalGradient = lossGradient + (regularizerGradient * (rho / 2.0))
    Vectors.fromBreeze(totalGradient)
  }
}

object SparseLogisticRegressionADMMUpdater {
  // to avoid numerical precision issues.
  private val maxAbsMargin = 10000

  def clampToRange(lower: Double = -maxAbsMargin, upper: Double = maxAbsMargin)(margin: Double): Double =
    math.min(upper, math.max(lower, margin))

  def logPhi(margin: Double): Double = {
    val t = clampToRange()(margin)
    if (t > 0) -math.log1p(math.exp(-t)) else t - math.log1p(math.exp(t))
  }

  def phi(margin: Double): Double = {
    val t = clampToRange()(margin)
    if (t > 0) 1.0 / (1 + math.exp(-t)) else math.exp(t) / (1 + math.exp(t))
  }
}

/**
 * Train a classification model for Logistic Regression using ADMM.
 * NOTE: Labels used in Logistic Regression should be {0, 1}
 */
class SparseLogisticRegressionWithADMM(
  val numIterations: Int,
  val lambda: Double,
  val rho: Double)
    extends GeneralizedLinearAlgorithm[LogisticRegressionModel]
    with Serializable {

  override val optimizer = new ADMMOptimizer(
    numIterations,
    new SparseLogisticRegressionADMMUpdater(lambda = lambda, rho = rho))

  // override val validators = List(DataValidators.classificationLabels)

  override def createModel(
    weights: Vector,
    intercept: Double): LogisticRegressionModel =
    new LogisticRegressionModel(weights, intercept)
}

/**
 * Top-level methods for calling Logistic Regression.
 * NOTE: Labels used in Logistic Regression should be {0, 1}
 */
object SparseLogisticRegressionWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    lambda: Double,
    rho: Double) = {
    new SparseLogisticRegressionWithADMM(numIterations, lambda, rho).run(input)
  }
}
