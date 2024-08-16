//STUDENT NAME: WoongKeol Kim
//STUDENT ID: 200050598

//Hero is at the position of agent P (variable), if agent P's position is identical to Hero's position 
at(P) :- pos(P,X,Y) & pos(hero,X,Y).


//Initial goal
!started.
!check(slots).

/*
* In the event that the agent must achieve "started", under all circumstances, print the message.
*/
+!started : true
   <- .print("I'm not scared of that smelly Goblin!").
   
// Couldn't collect all items
+!check(slots) : pos(hero, 7, 7)
	<- .print("Couldn't collect all items! Done!!").
	
// If the agent has all three items, then take the goblin and print "Done!!"
+!check(slots) : hero(gem) & hero(coin) & hero(vase)
	<- !take(goblin);
		.print("Done!!").
	
// If the agent does not have any of the three items, then move to the next slot and check the slots again.
+!check(slots) : not coin(hero) & not vase(hero) & not gem(hero)
	<- next(slot);
	!check(slots).
+!check(slots).
	
// If the agent has all three items, then move to the location of L, drop all the items.
+!take(L) : hero(gem) & hero(coin) & hero(vase)
	<- !at(L);
		drop(coin);
		drop(vase);
		drop(gem).
	
// If the slot of hero's position has the gem, then ensure the pick and check the slots again.
+!take(L) : gem(hero)
	<- 	!ensure_pick(G);
		!check(slots).
	
// If the slot of hero's position has the vase, then ensure the pick and check the slots again.
+!take(L) : vase(hero)
	<- 	!ensure_pick(V);
		!check(slots).
	
// If the slot of hero's position has the coin, then ensure the pick and check the slots again.
+!take(L) : coin(hero)
	<- 	!ensure_pick(C);
		!check(slots).
	
// If the slot of hero's position has the coin, then try to take it to goblin and print "There is coin"
+coin(hero) : true 
	<- .print("There is coin");
	!take(goblin).

// If the slot of hero's position has the vase, then try to take it to goblin and print "There is vase"
+vase(hero) : true
	<- .print("There is vase");
	!take(goblin).

// If the slot of hero's position has the gem, then try to take it to goblin and print "There is gem"
+gem(hero) : true
	<- .print("There is Gem");
	!take(goblin).
	
// Ensure the pick.
+!ensure_pick(C) : coin(hero)
	<- pick(coin);
	!ensure_pick(coin).

// Ensure the pick.
+!ensure_pick(V) : vase(hero)
	<- pick(vase);
	!ensure_pick(vase).

// Ensure the pick.
+!ensure_pick(G) : gem(hero)
	<- pick(gem);
	!ensure_pick(gem).

+!ensure_pick(_).

// If the agent is at L, then it is true that the agent is at L.
+!at(L):at(L).
+!at(L) <- ?pos(L,X,Y);
			move_towards(X,Y);
			!at(L).

