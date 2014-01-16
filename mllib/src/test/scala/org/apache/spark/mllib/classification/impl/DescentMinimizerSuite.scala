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

import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

object NewtonsMethod {
  def descentFunction: DescentFunction[Double] = {
    new DescentFunction[Double] {
      def descend(current: Double): Double = {
        val epsilon = 0.01
        current - (2 * current) * epsilon
      }
      def objective(current: Double): Double = {
        current * current
      }
    }
  }
}

class SVMWithADMMSuite
    extends FunSuite
    with BeforeAndAfterAll
    with ShouldMatchers {
  test("solves ax = 0") {
    val minimizer = DescentMinimizer[Double](100, 0.01, 0.001)

    minimizer.minimize(NewtonsMethod.descentFunction, 5) should equal (0.00)
  }
}
